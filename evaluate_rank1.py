"""
============================================================
EVALUATE_RANK1 - Accuracy rank-1/rank-5 (identificazione closed-set)
                  a partire dai checkpoint GIA' SALVATI dal nested_cv
============================================================

Non riaddestra nulla: per ogni outer fold (palm_outer_fold_XX.pt /
dorsal_outer_fold_XX.pt) ricarica il modello, ricostruisce il dataset
di test dello stesso fold (outer_test_subjects, salvato dentro al
checkpoint) ed estrae gli embedding con un semplice forward in eval().

Da questi embedding calcola:
  - EER open-set (stessa funzione compute_eer gia' usata in training,
    solo per controllo di coerenza col valore salvato nel checkpoint)
  - rank-1 / rank-5 identification accuracy (closed-set, leave-one-out
    dentro al singolo fold di test)

Se hai un solo checkpoint finale (non nested_cv) usa --mode single.

============================================================
USO
============================================================

# Tutti gli outer fold del PALMO (usa i soggetti di test salvati nel checkpoint)
python evaluate_rank1.py --stream palm \
    --data_dir dataset_preprocessed \
    --checkpoint_dir models_final/nested_cv \
    --mode nested_cv

# Tutti gli outer fold del DORSO
python evaluate_rank1.py --stream dorsal \
    --data_dir dataset_preprocessed \
    --checkpoint_dir models_final_dorsal/nested_cv \
    --mode nested_cv

# Un solo checkpoint (es. modello finale), test su un elenco di soggetti a scelta
python evaluate_rank1.py --stream palm \
    --data_dir dataset_preprocessed \
    --checkpoint models_final/palm_embedding_final.pt \
    --mode single \
    --eval_subjects_file test_subjects.txt

Se --eval_subjects_file non e' passato in modalita' 'single', vengono
usati TUTTI i soggetti presenti in --data_dir (attenzione: se sono
STATI USATI in training per quel checkpoint il numero non e' piu'
open-set/onesto, va bene solo come sanity check).
============================================================
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ============================================================
# RANK-1 / RANK-K (closed-set identification, leave-one-out nel fold)
# ============================================================
@torch.no_grad()
def compute_rank_k(embeddings: torch.Tensor, subject_ids: list[str], ranks=(1, 5)):
    """
    Closed-set identification accuracy.

    Per ogni campione i (probe), la gallery e' formata da TUTTI gli altri
    campioni dello stesso fold di test (leave-one-out): si cerca il
    soggetto corretto tra i piu' simili, escludendo il confronto con se
    stesso. Richiede almeno 2 campioni per soggetto per essere valutabile
    (un soggetto con un solo campione non ha mai un vero match possibile
    e viene escluso dal conteggio).

    Ritorna un dizionario tipo {"rank-1": 0.93, "rank-5": 0.99, "n_valid_probes": 412}
    """
    emb = F.normalize(embeddings, dim=1)
    sims = emb @ emb.T
    sims.fill_diagonal_(-1e9)  # niente self-match

    subj = np.array(subject_ids)
    n = len(subj)
    max_rank = max(ranks)
    correct_at = {r: 0 for r in ranks}
    n_valid = 0

    for i in range(n):
        if (subj == subj[i]).sum() < 2:
            continue  # soggetto senza altri campioni nel fold: non valutabile
        n_valid += 1
        k = min(max_rank, n - 1)
        top_idx = torch.topk(sims[i], k=k).indices.numpy()
        matched = np.where(subj[top_idx] == subj[i])[0]
        if len(matched) == 0:
            continue
        best_rank = matched[0] + 1
        for r in ranks:
            if best_rank <= r:
                correct_at[r] += 1

    result = {f"rank-{r}": correct_at[r] / max(1, n_valid) for r in ranks}
    result["n_valid_probes"] = n_valid
    result["n_total_samples"] = n
    return result


# ============================================================
# VERIFICATION ACCURACY (accuratezza binaria genuine/impostor alla
# soglia ottimale) - e' la metrica "accuracy" riportata dai paper che
# la presentano ACCOPPIATA all'EER (es. "accuracy 91.8%, EER 0.082%").
# Non va confusa col rank-1 (identification accuracy): qui il task e'
# binario (stesso soggetto si'/no su coppie), non un ranking su tutta
# la gallery.
# ============================================================
@torch.no_grad()
def compute_verification_accuracy(embeddings: torch.Tensor, subject_ids: list[str], core):
    """
    Riusa le stesse coppie genuine/impostor di compute_eer() (funzione
    build_verification_pairs, identica in palm_core/dorsal_core: 'core'
    e' il modulo gia' importato da evaluate_one_checkpoint), ma calcola
    l'accuratezza binaria (TP+TN)/totale alla soglia che massimizza
    l'accuratezza stessa (spesso vicina, ma non identica, alla soglia
    di EER: qui viene ricercata esplicitamente).
    """
    pos_pairs, neg_pairs = core.build_verification_pairs(subject_ids)
    if not pos_pairs or not neg_pairs:
        return None

    emb = F.normalize(embeddings, dim=1)
    sims = lambda pairs: (emb[[p[0] for p in pairs]] * emb[[p[1] for p in pairs]]).sum(1).cpu().numpy()
    genuine, impostor = sims(pos_pairs), sims(neg_pairs)

    thresholds = np.linspace(-1, 1, 500)
    accs = []
    for t in thresholds:
        tp = (genuine >= t).sum()
        tn = (impostor < t).sum()
        accs.append((tp + tn) / (len(genuine) + len(impostor)))
    accs = np.array(accs)
    best_idx = int(np.argmax(accs))

    return {
        "verification_accuracy": float(accs[best_idx]),
        "best_threshold": float(thresholds[best_idx]),
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
    }


# ============================================================
# ESTRAZIONE EMBEDDING (identica alla evaluate_open_set() di *_run.py,
# ma tiene anche gli embedding, non solo l'EER)
# ============================================================
@torch.no_grad()
def extract_embeddings(model, loader, device, stream: str):
    model.eval()
    all_emb, all_subj = [], []

    for batch in loader:
        if stream == "palm":
            emb = model(
                batch["palm_hand"].to(device, non_blocking=True),
                batch["palm_roi"].to(device, non_blocking=True),
                batch["knuckles"].to(device, non_blocking=True),
                batch["knuckle_mask"].to(device, non_blocking=True),
            )
        else:  # dorsal
            emb = model(
                batch["dorsal_hand"].to(device, non_blocking=True),
                batch["knuckles"].to(device, non_blocking=True),
                batch["knuckle_mask"].to(device, non_blocking=True),
            )
        all_emb.append(emb.cpu())
        all_subj.extend(batch["subject_id"])

    if not all_emb:
        return None, None
    return torch.cat(all_emb, dim=0), all_subj


# ============================================================
# CARICAMENTO MODELLO DA CHECKPOINT (nessun training coinvolto)
# ============================================================
def load_model_from_checkpoint(checkpoint_path, stream: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    n_knuckles = ckpt["n_knuckles"]
    embedding_dim = ckpt.get("embedding_dim")

    # Gli iperparametri di freeze non influenzano i pesi gia' salvati
    # (servono solo per costruire l'architettura giusta prima di
    # caricare lo state_dict), quindi va bene usare i default o quelli
    # eventualmente salvati in 'selected_hyperparameters'.
    hp = ckpt.get("selected_hyperparameters", {})
    freeze_backbone_texture = hp.get("freeze_vit", hp.get("freeze_swin", True))
    freeze_mobilenet = hp.get("freeze_mobilenet", False)

    if stream == "palm":
        from palm_core import cfg, PalmEmbeddingNet, ArcMarginHead
        if embedding_dim is None:
            embedding_dim = cfg.EMBEDDING_DIM
        model = PalmEmbeddingNet(
            n_knuckles=n_knuckles, embedding_dim=embedding_dim,
            freeze_vit=freeze_backbone_texture, freeze_mobilenet=freeze_mobilenet,
        ).to(device)
    else:
        from dorsal_core import cfg, DorsalEmbeddingNet, ArcMarginHead
        if embedding_dim is None:
            embedding_dim = cfg.EMBEDDING_DIM
        model = DorsalEmbeddingNet(
            n_knuckles=n_knuckles, embedding_dim=embedding_dim,
            freeze_swin=freeze_backbone_texture, freeze_mobilenet=freeze_mobilenet,
        ).to(device)

    n_classes = len(ckpt["subject_to_label"])
    head = ArcMarginHead(embedding_dim, n_classes).to(device)  # non serve in eval, ma load_state_dict lo richiede

    model.load_state_dict(ckpt["model_state"])
    head.load_state_dict(ckpt["head_state"])
    model.eval()

    return model, ckpt


def build_eval_dataset(stream: str, data_dir, eval_subjects, n_knuckles_max,
                        embed_cache_path=None):
    if stream == "palm":
        from palm_core import PalmBiometricDataset
        vit_embed_cache = None
        if embed_cache_path:
            data = np.load(embed_cache_path, allow_pickle=True)
            vit_embed_cache = {str(p): data["embeds"][i] for i, p in enumerate(data["paths"])}
        ds = PalmBiometricDataset(
            data_dir, subject_ids=eval_subjects, train=False,
            vit_embed_cache=vit_embed_cache,
        )
    else:
        from dorsal_core import DorsalBiometricDataset
        swin_embed_cache = None
        if embed_cache_path:
            data = np.load(embed_cache_path, allow_pickle=True)
            swin_embed_cache = {str(p): data["embeds"][i] for i, p in enumerate(data["paths"])}
        ds = DorsalBiometricDataset(
            data_dir, subject_ids=eval_subjects, train=False,
            swin_embed_cache=swin_embed_cache,
        )
    # stessa dimensionalita' con cui e' stato allenato il modello
    ds.n_knuckles_max = n_knuckles_max
    return ds


# ============================================================
# UN SINGOLO FOLD / CHECKPOINT
# ============================================================
def evaluate_one_checkpoint(checkpoint_path, stream: str, data_dir, device,
                             batch_size=32, num_workers=4, embed_cache_path=None,
                             eval_subjects_override=None):
    from importlib import import_module
    core = import_module("palm_core" if stream == "palm" else "dorsal_core")

    model, ckpt = load_model_from_checkpoint(checkpoint_path, stream, device)

    eval_subjects = eval_subjects_override or ckpt.get("outer_test_subjects")
    if eval_subjects is None:
        raise ValueError(
            f"{checkpoint_path}: nessun 'outer_test_subjects' nel checkpoint e "
            f"nessun --eval_subjects_file passato. Specifica i soggetti di test."
        )

    ds = build_eval_dataset(
        stream, data_dir, eval_subjects, ckpt["n_knuckles"], embed_cache_path
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    embeddings, subject_ids = extract_embeddings(model, loader, device, stream)
    if embeddings is None:
        raise RuntimeError(f"{checkpoint_path}: dataset di valutazione vuoto")

    eer_result = core.compute_eer(embeddings, subject_ids)
    verif_acc_result = compute_verification_accuracy(embeddings, subject_ids, core)
    rank_result = compute_rank_k(embeddings, subject_ids, ranks=(1, 5))

    return {
        "checkpoint": str(checkpoint_path),
        "outer_fold": ckpt.get("outer_fold"),
        "n_test_subjects": len(set(subject_ids)),
        "n_test_samples": len(subject_ids),
        "eer_recomputed": eer_result,
        "eer_saved_in_checkpoint": ckpt.get("best_eer"),
        "verification_accuracy": verif_acc_result,   # accuracy binaria genuine/impostor (accoppiata a EER)
        **rank_result,                                 # rank-1 / rank-5 = identification accuracy
    }


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Calcola rank-1/rank-5 (e ricontrolla l'EER) da checkpoint gia' salvati"
    )
    parser.add_argument("--stream", choices=["palm", "dorsal"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mode", choices=["nested_cv", "single"], default="nested_cv")

    # modalita' nested_cv: legge tutti i checkpoint di outer fold in una cartella
    parser.add_argument(
        "--checkpoint_dir", default=None,
        help="cartella con palm_outer_fold_XX.pt / dorsal_outer_fold_XX.pt "
             "(es. models_final/nested_cv oppure models_final_dorsal/nested_cv)",
    )

    # modalita' single: un solo checkpoint
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--eval_subjects_file", default=None,
        help="file di testo con un subject_id per riga; se omesso in modalita' "
             "'single' usa outer_test_subjects se presente nel checkpoint, "
             "altrimenti TUTTI i soggetti in --data_dir",
    )

    parser.add_argument(
        "--embed_cache_path", default=None,
        help="opzionale: .npz di embedding precalcolati (swin/vit) coerente "
             "con quello usato in training, per velocizzare l'estrazione",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_json", default=None, help="dove salvare il riepilogo JSON")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_results = []

    if args.mode == "nested_cv":
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir richiesto in modalita' nested_cv")
        prefix = "palm_outer_fold_" if args.stream == "palm" else "dorsal_outer_fold_"
        ckpts = sorted(Path(args.checkpoint_dir).glob(f"{prefix}*.pt"))
        if not ckpts:
            raise FileNotFoundError(
                f"Nessun checkpoint '{prefix}*.pt' trovato in {args.checkpoint_dir}"
            )
        print(f"Trovati {len(ckpts)} checkpoint di outer fold in {args.checkpoint_dir}")
        for ckpt_path in ckpts:
            print(f"\n--- {ckpt_path.name} ---")
            res = evaluate_one_checkpoint(
                ckpt_path, args.stream, args.data_dir, device,
                batch_size=args.batch_size, num_workers=args.num_workers,
                embed_cache_path=args.embed_cache_path,
            )
            print(json.dumps(res, indent=2, default=str))
            fold_results.append(res)

    else:  # single
        if not args.checkpoint:
            raise ValueError("--checkpoint richiesto in modalita' single")
        eval_subjects_override = None
        if args.eval_subjects_file:
            eval_subjects_override = [
                line.strip() for line in Path(args.eval_subjects_file).read_text().splitlines()
                if line.strip()
            ]
        else:
            # fallback: se il checkpoint non ha outer_test_subjects, usa tutto data_dir
            from importlib import import_module
            core = import_module("palm_core" if args.stream == "palm" else "dorsal_core")
            ckpt_probe = torch.load(args.checkpoint, map_location="cpu")
            if "outer_test_subjects" not in ckpt_probe:
                eval_subjects_override = core.list_subjects(args.data_dir)
                print(
                    "ATTENZIONE: nessun outer_test_subjects nel checkpoint e nessun "
                    "--eval_subjects_file passato: uso TUTTI i soggetti di --data_dir. "
                    "Se questo checkpoint e' stato allenato su (parte di) questi "
                    "soggetti, il risultato NON e' open-set/onesto."
                )

        res = evaluate_one_checkpoint(
            args.checkpoint, args.stream, args.data_dir, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
            embed_cache_path=args.embed_cache_path,
            eval_subjects_override=eval_subjects_override,
        )
        print(json.dumps(res, indent=2, default=str))
        fold_results.append(res)

    # ============================================================
    # RIEPILOGO AGGREGATO (media +- std sugli outer fold, come per l'EER)
    # ============================================================
    if len(fold_results) > 1:
        rank1 = np.array([r["rank-1"] for r in fold_results])
        rank5 = np.array([r["rank-5"] for r in fold_results])
        eer = np.array([r["eer_recomputed"]["eer"] for r in fold_results if r["eer_recomputed"]])
        verif_acc = np.array([
            r["verification_accuracy"]["verification_accuracy"]
            for r in fold_results if r["verification_accuracy"]
        ])

        summary = {
            "stream": args.stream,
            "n_folds": len(fold_results),
            "mean_rank1": float(rank1.mean()), "std_rank1": float(rank1.std()),
            "mean_rank5": float(rank5.mean()), "std_rank5": float(rank5.std()),
            "mean_eer": float(eer.mean()) if len(eer) else None,
            "std_eer": float(eer.std()) if len(eer) else None,
            "mean_verification_accuracy": float(verif_acc.mean()) if len(verif_acc) else None,
            "std_verification_accuracy": float(verif_acc.std()) if len(verif_acc) else None,
        }
        print("\n" + "=" * 60)
        print("RIEPILOGO NESTED CV")
        print("=" * 60)
        print(json.dumps(summary, indent=2))
        fold_results = {"folds": fold_results, "summary": summary}

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(fold_results, indent=2, default=str))
        print(f"\nRisultati salvati in: {args.output_json}")


if __name__ == "__main__":
    main()