"""
============================================================
DORSAL_RUN - Training + Nested K-Fold + Inferenza
============================================================

Comandi principali:

# Training semplice (split train/validation per soggetto)
python dorsal_run.py train --data_dir dataset_preprocessed --epochs 60

# Nested K-Fold:
# Outer K-Fold = valutazione finale
# Inner K-Fold = selezione iperparametri
python dorsal_run.py nested_cv --data_dir dataset_preprocessed \
    --outer_folds 5 --inner_folds 4 \
    --inner_epochs 25 --outer_epochs 60 \
    --lr_grid 0.0001,0.0003 --freeze_mobilenet_grid false,true

# Verifica 1:1
python dorsal_run.py verify --checkpoint models_final_dorsal/dorsal_embedding_best.pt \
    --base1 dataset_preprocessed/0001/0001_dorsal_001 \
    --base2 dataset_preprocessed/0001/0001_dorsal_002
============================================================
"""

from __future__ import annotations
import argparse
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dorsal_core import (
    cfg, dorsal_hand_transform, knuckle_transform,
    DorsalEmbeddingNet, ArcMarginHead,
    DorsalBiometricDataset, split_subjects, compute_eer,
    list_subjects, kfold_subject_splits,
)


# ============================================================
# UTILITA'
# ============================================================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Booleano non valido: {value}")


def parse_float_grid(value: str):
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("La griglia non puo' essere vuota")
    return vals


def parse_bool_grid(value: str):
    vals = [parse_bool(x) for x in value.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("La griglia non puo' essere vuota")
    return vals


def make_loader(dataset, batch_size, shuffle=False, drop_last=False):
    # drop_last solo se ci sono almeno batch_size campioni.
    effective_drop_last = drop_last and len(dataset) >= batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        drop_last=effective_drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def save_checkpoint(path, epoch, model, head, optimizer, scheduler,
                    best_eer, n_knuckles, subject_to_label, extra=None):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "head_state": head.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_eer": best_eer,
        "n_knuckles": n_knuckles,
        "subject_to_label": subject_to_label,
        "embedding_dim": cfg.EMBEDDING_DIM,
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, model, head, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    head.load_state_dict(ckpt["head_state"])
    if optimizer is not None and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


# ============================================================
# TRAIN / EVALUATE
# ============================================================
def train_one_epoch(model, head, loader, optimizer, criterion, device, epoch, log_every=20):
    model.train()
    head.train()

    total_loss, total_correct, total_n = 0.0, 0, 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        dorsal_hand = batch["dorsal_hand"].to(device, non_blocking=True)
        knuckles = batch["knuckles"].to(device, non_blocking=True)
        knuckle_mask = batch["knuckle_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        emb = model(dorsal_hand, knuckles, knuckle_mask)
        logits = head(emb, labels)
        loss = criterion(logits, labels)
        loss.backward()

        trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += labels.size(0)

        if step % log_every == 0:
            print(
                f"  [epoch {epoch}] step {step}/{len(loader)} "
                f"loss={loss.item():.4f} acc={total_correct/max(1,total_n):.4f}"
            )

    return (
        total_loss / max(1, total_n),
        total_correct / max(1, total_n),
        time.time() - t0,
    )


@torch.no_grad()
def evaluate_open_set(model, loader, device):
    model.eval()
    all_emb, all_subj = [], []

    for batch in loader:
        emb = model(
            batch["dorsal_hand"].to(device, non_blocking=True),
            batch["knuckles"].to(device, non_blocking=True),
            batch["knuckle_mask"].to(device, non_blocking=True),
        )
        all_emb.append(emb.cpu())
        all_subj.extend(batch["subject_id"])

    if not all_emb:
        return None
    return compute_eer(torch.cat(all_emb, dim=0), all_subj)


def build_datasets(data_dir, train_subjects, eval_subjects=None,
                   n_knuckles_max=None):
    train_ds = DorsalBiometricDataset(
        data_dir, subject_ids=train_subjects, train=True
    )
    eval_ds = None
    if eval_subjects is not None:
        eval_ds = DorsalBiometricDataset(
            data_dir, subject_ids=eval_subjects, train=False
        )

    if n_knuckles_max is None:
        candidates = [train_ds.n_knuckles_max]
        if eval_ds is not None:
            candidates.append(eval_ds.n_knuckles_max)
        n_knuckles_max = max(candidates)

    # Stessa dimensionalita' architetturale per tutti i dataset del fold.
    train_ds.n_knuckles_max = n_knuckles_max
    if eval_ds is not None:
        eval_ds.n_knuckles_max = n_knuckles_max

    return train_ds, eval_ds, n_knuckles_max


def init_csv_logger(path):
    """Crea/apre un CSV di log epoca-per-epoca. Ritorna una funzione log_row(dict)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    f = open(path, "a", newline="", encoding="utf-8")
    import csv
    fieldnames = ["epoch", "train_loss", "train_acc", "eval_eer",
                  "eval_eer_threshold", "lr", "epoch_seconds", "timestamp"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()

    def log_row(row):
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()

    return log_row, f


def train_model(train_subjects, data_dir, epochs, batch_size, lr,
                freeze_swin, freeze_mobilenet, seed,
                eval_subjects=None, verbose=True, select_best_on_eval=True,
                log_csv_path=None):
    """
    Addestra un modello su train_subjects.

    Se eval_subjects e' presente, restituisce anche l'EER open-set.
    Questa funzione e' riutilizzata sia dal training semplice sia
    dall'inner/outer loop del Nested K-Fold.
    """
    set_seed(seed)
    device = cfg.DEVICE

    train_ds, eval_ds, n_knuckles = build_datasets(
        data_dir, train_subjects, eval_subjects
    )
    if len(train_ds) == 0:
        raise RuntimeError("Dataset di training vuoto per questo fold")

    train_loader = make_loader(train_ds, batch_size, shuffle=True, drop_last=True)
    eval_loader = (
        make_loader(eval_ds, batch_size, shuffle=False)
        if eval_ds is not None and len(eval_ds) > 0 else None
    )

    model = DorsalEmbeddingNet(
        n_knuckles,
        cfg.EMBEDDING_DIM,
        freeze_swin=freeze_swin,
        freeze_mobilenet=freeze_mobilenet,
    ).to(device)

    head = ArcMarginHead(cfg.EMBEDDING_DIM, train_ds.num_classes).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=lr, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    criterion = nn.CrossEntropyLoss()

    best_eer = float("inf")
    best_epoch = 0
    best_model_state = None
    best_head_state = None

    log_row, log_file = (None, None)
    if log_csv_path is not None:
        log_row, log_file = init_csv_logger(log_csv_path)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, dt = train_one_epoch(
            model, head, train_loader, optimizer, criterion, device, epoch
        )
        scheduler.step()

        metrics = None
        if eval_loader is not None and select_best_on_eval:
            metrics = evaluate_open_set(model, eval_loader, device)
            if metrics is not None and metrics["eer"] < best_eer:
                best_eer = metrics["eer"]
                best_epoch = epoch
                best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                best_head_state = {
                    k: v.detach().cpu().clone()
                    for k, v in head.state_dict().items()
                }

        if verbose:
            msg = (
                f"Epoch {epoch}/{epochs} - loss={train_loss:.4f} "
                f"acc={train_acc:.4f} ({dt:.1f}s) "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if metrics:
                msg += (
                    f" | eval EER={metrics['eer']*100:.2f}% "
                    f"@thr={metrics['eer_threshold']:.3f}"
                )
            print(msg)

        if log_row is not None:
            import datetime
            log_row({
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "eval_eer": metrics["eer"] if metrics else "",
                "eval_eer_threshold": metrics["eer_threshold"] if metrics else "",
                "lr": scheduler.get_last_lr()[0], "epoch_seconds": round(dt, 2),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            })

    if log_file is not None:
        log_file.close()

    # Con validation ripristiniamo l'epoca migliore.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        head.load_state_dict(best_head_state)
        final_metrics = evaluate_open_set(model, eval_loader, device)
    elif eval_loader is not None:
        # Nel refit outer l'eval set puo' essere l'outer test:
        # viene valutato UNA SOLA VOLTA qui, senza influenzare checkpoint/epoche.
        final_metrics = evaluate_open_set(model, eval_loader, device)
        best_epoch = epochs
        best_eer = final_metrics["eer"] if final_metrics is not None else None
    else:
        final_metrics = None
        best_epoch = epochs
        best_eer = None

    return {
        "model": model,
        "head": head,
        "train_ds": train_ds,
        "n_knuckles": n_knuckles,
        "metrics": final_metrics,
        "best_epoch": best_epoch,
        "best_eer": best_eer,
    }


# ============================================================
# TRAINING SEMPLICE (compatibilita')
# ============================================================
def cmd_train(args):
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Device: {device}")

    train_subjects, val_subjects = split_subjects(args.data_dir)
    print(
        f"Soggetti train: {len(train_subjects)} | "
        f"Soggetti validation (open-set): {len(val_subjects)}"
    )

    result = train_model(
        train_subjects=train_subjects,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_swin=args.freeze_swin,
        freeze_mobilenet=args.freeze_mobilenet,
        seed=cfg.SEED,
        eval_subjects=val_subjects,
        verbose=True,
        log_csv_path=cfg.FINAL_MODEL_DIR / "train_log.csv",
    )

    model = result["model"]
    head = result["head"]
    train_ds = result["train_ds"]
    best_eer = result["metrics"]["eer"] if result["metrics"] else float("nan")

    final_path = cfg.FINAL_MODEL_DIR / "dorsal_embedding_final.pt"
    save_checkpoint(
        final_path, result["best_epoch"], model, head,
        optimizer=None, scheduler=None, best_eer=best_eer,
        n_knuckles=result["n_knuckles"],
        subject_to_label=train_ds.subject_to_label,
        extra={"training_mode": "simple_subject_split"},
    )

    print(f"\nTraining completato. Modello finale: {final_path}")
    if result["metrics"]:
        print(
            f"EER validation open-set: {result['metrics']['eer']*100:.2f}% "
            f"@thr={result['metrics']['eer_threshold']:.3f}"
        )


# ============================================================
# NESTED K-FOLD
# ============================================================
def make_hyperparameter_grid(args):
    grid = []
    for lr, freeze_swin, freeze_mobilenet in itertools.product(
        args.lr_grid,
        args.freeze_swin_grid,
        args.freeze_mobilenet_grid,
    ):
        grid.append({
            "lr": float(lr),
            "freeze_swin": bool(freeze_swin),
            "freeze_mobilenet": bool(freeze_mobilenet),
        })
    return grid


def inner_model_selection(outer_train_subjects, args, outer_fold):
    """
    INNER LOOP:
    - nessun soggetto dell'outer test entra qui;
    - per ogni combinazione di iperparametri esegue Inner K-Fold;
    - seleziona la configurazione con EER medio piu' basso.
    """
    candidates = make_hyperparameter_grid(args)
    candidate_results = []

    print(f"\n[OUTER {outer_fold}] INNER MODEL SELECTION")
    print(
        f"  Soggetti disponibili per inner CV: {len(outer_train_subjects)} | "
        f"candidati: {len(candidates)}"
    )

    for cand_idx, hp in enumerate(candidates, start=1):
        print(
            f"\n  Candidato {cand_idx}/{len(candidates)}: "
            f"lr={hp['lr']} freeze_swin={hp['freeze_swin']} "
            f"freeze_mobilenet={hp['freeze_mobilenet']}"
        )

        fold_eers = []
        fold_epochs = []

        for inner_fold, inner_train, inner_val in kfold_subject_splits(
            outer_train_subjects,
            args.inner_folds,
            seed=cfg.SEED + outer_fold * 1000 + cand_idx,
        ):
            print(
                f"    [Inner {inner_fold}/{args.inner_folds}] "
                f"train_subjects={len(inner_train)} "
                f"val_subjects={len(inner_val)}"
            )

            result = train_model(
                train_subjects=inner_train,
                data_dir=args.data_dir,
                epochs=args.inner_epochs,
                batch_size=args.batch_size,
                lr=hp["lr"],
                freeze_swin=hp["freeze_swin"],
                freeze_mobilenet=hp["freeze_mobilenet"],
                seed=cfg.SEED + outer_fold * 10000 + cand_idx * 100 + inner_fold,
                eval_subjects=inner_val,
                verbose=args.verbose_inner,
                log_csv_path=cfg.FINAL_MODEL_DIR / "nested_cv" /
                    f"train_log_outer{outer_fold:02d}_cand{cand_idx}_inner{inner_fold}.csv",
            )

            metrics = result["metrics"]
            if metrics is None:
                raise RuntimeError(
                    "Impossibile calcolare EER nell'inner fold. "
                    "Ogni validation fold deve contenere abbastanza campioni per soggetto."
                )

            fold_eers.append(metrics["eer"])
            fold_epochs.append(result["best_epoch"])

            print(
                f"      -> Inner EER={metrics['eer']*100:.2f}% "
                f"@thr={metrics['eer_threshold']:.3f} "
                f"(best epoch={result['best_epoch']})"
            )

            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_eer = float(np.mean(fold_eers))
        std_eer = float(np.std(fold_eers))
        mean_epoch = max(1, int(round(np.mean(fold_epochs))))

        candidate_results.append({
            **hp,
            "mean_eer": mean_eer,
            "std_eer": std_eer,
            "fold_eers": fold_eers,
            "selected_epochs": fold_epochs,
            "mean_best_epoch": mean_epoch,
        })

        print(
            f"  ==> Candidato {cand_idx}: mean EER={mean_eer*100:.2f}% "
            f"+/- {std_eer*100:.2f}% | epoch finale suggerita={mean_epoch}"
        )

    # Tie-break: prima EER medio, poi deviazione standard.
    candidate_results.sort(key=lambda x: (x["mean_eer"], x["std_eer"]))
    best = candidate_results[0]

    print(
        f"\n  >>> BEST OUTER {outer_fold}: "
        f"lr={best['lr']} freeze_swin={best['freeze_swin']} "
        f"freeze_mobilenet={best['freeze_mobilenet']} "
        f"| inner mean EER={best['mean_eer']*100:.2f}%"
    )

    return best, candidate_results


def cmd_nested_cv(args):
    """
    Protocollo Nested K-Fold:

    OUTER:
        train+validation -----------------> selezione modello nell'INNER
        test (soggetti mai visti) --------> una sola valutazione finale

    INNER:
        train/validation per soggetto ----> scelta iperparametri

    IMPORTANTE: l'outer test fold non viene mai usato per scegliere
    iperparametri, epoche o checkpoint.
    """
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    all_subjects = list_subjects(args.data_dir)

    if len(all_subjects) < args.outer_folds:
        raise ValueError(
            f"Soggetti ({len(all_subjects)}) < outer_folds ({args.outer_folds})"
        )

    # Dopo aver rimosso l'outer test deve essere ancora possibile fare l'inner CV.
    min_outer_train = len(all_subjects) - int(np.ceil(len(all_subjects) / args.outer_folds))
    if min_outer_train < args.inner_folds:
        raise ValueError(
            f"Con {len(all_subjects)} soggetti, outer_folds={args.outer_folds} "
            f"lascia troppo pochi soggetti per inner_folds={args.inner_folds}."
        )

    output_dir = cfg.FINAL_MODEL_DIR / "nested_cv"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("NESTED K-FOLD - DORSAL BIOMETRIC EMBEDDING")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Soggetti totali: {len(all_subjects)}")
    print(f"Outer folds: {args.outer_folds}")
    print(f"Inner folds: {args.inner_folds}")
    print(f"Inner epochs: {args.inner_epochs}")
    print(f"Outer epochs: {args.outer_epochs}")
    print(f"Output: {output_dir}")

    outer_results = []

    for outer_fold, outer_train, outer_test in kfold_subject_splits(
        all_subjects, args.outer_folds, seed=cfg.SEED
    ):
        print("\n" + "#" * 72)
        print(
            f"OUTER FOLD {outer_fold}/{args.outer_folds} | "
            f"train+inner={len(outer_train)} | final test={len(outer_test)}"
        )
        print("#" * 72)

        # 1) INNER CV: selezione iperparametri.
        best_hp, all_candidates = inner_model_selection(
            outer_train, args, outer_fold
        )

        # 2) Refit sul 100% dei soggetti dell'outer training set.
        #    Il numero di epoche viene dalla media delle best epoch dell'inner CV.
        final_epochs = (
            args.outer_epochs if args.outer_epochs is not None
            else best_hp["mean_best_epoch"]
        )

        print(
            f"\n[OUTER {outer_fold}] REFIT finale su {len(outer_train)} soggetti "
            f"per {final_epochs} epoche"
        )

        final_result = train_model(
            train_subjects=outer_train,
            data_dir=args.data_dir,
            epochs=final_epochs,
            batch_size=args.batch_size,
            lr=best_hp["lr"],
            freeze_swin=best_hp["freeze_swin"],
            freeze_mobilenet=best_hp["freeze_mobilenet"],
            seed=cfg.SEED + outer_fold * 99999,
            eval_subjects=outer_test,  # SOLO valutazione finale
            verbose=True,
            select_best_on_eval=False,   # l'outer test NON influenza il training
            log_csv_path=cfg.FINAL_MODEL_DIR / "nested_cv" / f"train_log_outer{outer_fold:02d}_refit.csv",
        )

        test_metrics = final_result["metrics"]
        if test_metrics is None:
            raise RuntimeError(f"Impossibile calcolare EER sull'outer fold {outer_fold}")

        fold_path = output_dir / f"dorsal_outer_fold_{outer_fold:02d}.pt"
        save_checkpoint(
            fold_path,
            final_result["best_epoch"],
            final_result["model"],
            final_result["head"],
            optimizer=None,
            scheduler=None,
            best_eer=test_metrics["eer"],
            n_knuckles=final_result["n_knuckles"],
            subject_to_label=final_result["train_ds"].subject_to_label,
            extra={
                "training_mode": "nested_kfold_outer",
                "outer_fold": outer_fold,
                "outer_folds": args.outer_folds,
                "inner_folds": args.inner_folds,
                "selected_hyperparameters": best_hp,
                "outer_test_subjects": outer_test,
            },
        )

        fold_result = {
            "outer_fold": outer_fold,
            "n_outer_train_subjects": len(outer_train),
            "n_outer_test_subjects": len(outer_test),
            "selected_hyperparameters": best_hp,
            "all_inner_candidates": all_candidates,
            "outer_test_metrics": test_metrics,
            "checkpoint": str(fold_path),
        }
        outer_results.append(fold_result)

        print(
            f"\n[OUTER {outer_fold}] FINAL TEST (mai usato nell'inner CV): "
            f"EER={test_metrics['eer']*100:.2f}% "
            f"@thr={test_metrics['eer_threshold']:.3f}"
        )

        del final_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eers = np.array([x["outer_test_metrics"]["eer"] for x in outer_results])
    thresholds = np.array(
        [x["outer_test_metrics"]["eer_threshold"] for x in outer_results]
    )

    summary = {
        "protocol": {
            "name": "Nested K-Fold subject-disjoint",
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "seed": cfg.SEED,
            "selection_metric": "open-set EER",
            "unit_of_split": "subject_id",
        },
        "n_subjects": len(all_subjects),
        "outer_folds": outer_results,
        "summary": {
            "mean_outer_eer": float(eers.mean()),
            "std_outer_eer": float(eers.std()),
            "min_outer_eer": float(eers.min()),
            "max_outer_eer": float(eers.max()),
            "mean_eer_threshold": float(thresholds.mean()),
        },
    }

    summary_path = output_dir / "nested_cv_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("RISULTATO FINALE NESTED K-FOLD")
    print("=" * 72)
    print(
        f"Outer-test EER medio: {eers.mean()*100:.2f}% "
        f"+/- {eers.std()*100:.2f}%"
    )
    print(f"Min EER: {eers.min()*100:.2f}% | Max EER: {eers.max()*100:.2f}%")
    print(f"Threshold EER medio (solo descrittivo): {thresholds.mean():.3f}")
    print(f"Summary JSON: {summary_path}")


# ============================================================
# INFERENZA / VERIFICA
# ============================================================
class DorsalVerifier:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or cfg.DEVICE
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.n_knuckles = ckpt["n_knuckles"]
        self.model = DorsalEmbeddingNet(
            self.n_knuckles,
            ckpt.get("embedding_dim", cfg.EMBEDDING_DIM),
            freeze_swin=True,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.hand_tf = dorsal_hand_transform(train=False)
        self.knuckle_tf = knuckle_transform(train=False)
        print(
            f"Modello caricato da {checkpoint_path} "
            f"(epoch {ckpt['epoch']}, best_eer={ckpt.get('best_eer', float('nan')):.4f})"
        )

    def _load_sample(self, base_path):
        base_path = Path(base_path)
        folder, prefix = base_path.parent, base_path.name
        hand_path = folder / f"{prefix}_dorsal_hand.png"
        knuckle_paths = [
            p for p in sorted(folder.glob(f"{prefix}_dorsal_*.png"))
            if p.name != hand_path.name
        ]

        if not hand_path.exists():
            raise FileNotFoundError(
                f"File mancante per {base_path} (dorsal_hand)"
            )

        dorsal_hand = self.hand_tf(
            cv2.cvtColor(cv2.imread(str(hand_path)), cv2.COLOR_BGR2RGB)
        ).unsqueeze(0)

        knuckles = torch.zeros(
            1, self.n_knuckles, 3, *cfg.DORSAL_KNUCKLE_SIZE
        )
        mask = torch.zeros(1, self.n_knuckles)

        for i, kp in enumerate(knuckle_paths[:self.n_knuckles]):
            img = cv2.cvtColor(cv2.imread(str(kp)), cv2.COLOR_BGR2RGB)
            knuckles[0, i] = self.knuckle_tf(img)
            mask[0, i] = 1.0

        return (
            dorsal_hand.to(self.device),
            knuckles.to(self.device),
            mask.to(self.device),
        )

    @torch.no_grad()
    def embed(self, base_path) -> torch.Tensor:
        dorsal_hand, knuckles, mask = self._load_sample(base_path)
        return self.model(
            dorsal_hand, knuckles, mask
        ).squeeze(0).cpu()

    @torch.no_grad()
    def verify(self, base_path1, base_path2,
               threshold: float = cfg.VERIFICATION_THRESHOLD):
        emb1 = self.embed(base_path1)
        emb2 = self.embed(base_path2)
        similarity = F.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()
        return {
            "same_subject": bool(similarity >= threshold),
            "similarity": similarity,
            "threshold": threshold,
        }

    @torch.no_grad()
    def identify(self, probe_base_path, gallery: dict[str, torch.Tensor],
                 threshold: float = cfg.VERIFICATION_THRESHOLD):
        probe_emb = self.embed(probe_base_path)
        best_subject, best_sim = None, -1.0

        for subject_id, gallery_emb in gallery.items():
            sim = F.cosine_similarity(
                probe_emb.unsqueeze(0), gallery_emb.unsqueeze(0)
            ).item()
            if sim > best_sim:
                best_subject, best_sim = subject_id, sim

        if best_sim < threshold:
            return {"identified_subject": None, "similarity": best_sim}
        return {"identified_subject": best_subject, "similarity": best_sim}


def cmd_verify(args):
    verifier = DorsalVerifier(args.checkpoint)
    result = verifier.verify(
        args.base1, args.base2, threshold=args.threshold
    )
    print(json.dumps(result, indent=2))
    verdict = (
        "STESSO SOGGETTO"
        if result["same_subject"] else "SOGGETTI DIVERSI"
    )
    print(
        f"\n>>> {verdict}  (similarity={result['similarity']:.4f}, "
        f"soglia={result['threshold']:.4f})"
    )


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Training + Nested K-Fold + inferenza embedding biometrico dorso"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Training semplice.
    p_train = sub.add_parser(
        "train", help="addestra con semplice split train/validation per soggetto"
    )
    p_train.add_argument("--data_dir", default=str(cfg.DATA_DIR))
    p_train.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p_train.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p_train.add_argument("--lr", type=float, default=cfg.LR)
    p_train.add_argument("--freeze_swin", type=parse_bool, default=True)
    p_train.add_argument("--freeze_mobilenet", type=parse_bool, default=False)
    p_train.set_defaults(func=cmd_train)

    # Nested K-Fold.
    p_nested = sub.add_parser(
        "nested_cv",
        help="Nested K-Fold subject-disjoint: inner tuning + outer final test",
    )
    p_nested.add_argument("--data_dir", default=str(cfg.DATA_DIR))
    p_nested.add_argument("--outer_folds", type=int, default=cfg.OUTER_FOLDS)
    p_nested.add_argument("--inner_folds", type=int, default=cfg.INNER_FOLDS)
    p_nested.add_argument("--inner_epochs", type=int, default=cfg.INNER_EPOCHS)
    p_nested.add_argument(
        "--outer_epochs", type=int, default=cfg.OUTER_EPOCHS,
        help="epoche del refit outer dopo la selezione inner",
    )
    p_nested.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p_nested.add_argument(
        "--lr_grid", type=parse_float_grid, default=[cfg.LR],
        help="es. 0.0001,0.0003",
    )
    p_nested.add_argument(
        "--freeze_swin_grid", type=parse_bool_grid, default=[True],
        help="es. true,false",
    )
    p_nested.add_argument(
        "--freeze_mobilenet_grid", type=parse_bool_grid, default=[False, True],
        help="es. false,true",
    )
    p_nested.add_argument(
        "--verbose_inner", action="store_true",
        help="stampa ogni epoca anche durante tutti gli inner fold",
    )
    p_nested.set_defaults(func=cmd_nested_cv)

    # Verifica.
    p_verify = sub.add_parser(
        "verify", help="verifica 1:1 tra due acquisizioni preprocessate"
    )
    p_verify.add_argument("--checkpoint", required=True)
    p_verify.add_argument("--base1", required=True)
    p_verify.add_argument("--base2", required=True)
    p_verify.add_argument(
        "--threshold", type=float, default=cfg.VERIFICATION_THRESHOLD
    )
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()