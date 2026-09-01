"""
============================================================
MULTIMODAL_FUSION - Fusione Palmo + Dorso e Test su Nuovi Campioni
============================================================
Architettura di fusione a 3 livelli, coerente con palm_core/dorsal_core:

  1) FUSIONE INTRA-STREAM PALMO   -> gia' dentro PalmEmbeddingNet
     (texture + roi centrale + nocche FRIT -> palm_embedding, 256-d)
  2) FUSIONE INTRA-STREAM DORSO   -> gia' dentro DorsalEmbeddingNet
     (texture/vene Swin + nocche MobileNetV3 -> dorsal_embedding, 256-d)
  3) FUSIONE FINALE (macro-stream) -> SCORE-LEVEL FUSION
     similarity_finale = alpha * cos_sim(palm) + (1-alpha) * cos_sim(dorso)
     alpha calibrato sul validation set minimizzando l'EER fuso
     (piu' robusto del feature-level fusion con pochi campioni/soggetto
     e non richiede ri-training dei due modelli gia' addestrati).

Comandi:

# 1) Calibrazione alpha + raccolta metriche (palmo, dorso, fuso) su un set di validazione
python multimodal_fusion.py calibrate \
    --palm_checkpoint models_final/palm_embedding_final.pt \
    --dorsal_checkpoint models_final_dorsal/dorsal_embedding_final.pt \
    --data_dir dataset_preprocessed \
    --out_dir metrics_fusion

# 2) Verifica 1:1 multimodale su nuovi campioni (usa l'alpha calibrato)
python multimodal_fusion.py verify \
    --palm_checkpoint models_final/palm_embedding_final.pt \
    --dorsal_checkpoint models_final_dorsal/dorsal_embedding_final.pt \
    --alpha 0.55 \
    --base1 dataset_preprocessed/0001/0001_001 \
    --base2 dataset_preprocessed/0002/0002_003

# 3) Identificazione 1:N su una gallery
python multimodal_fusion.py identify \
    --palm_checkpoint ... --dorsal_checkpoint ... --alpha 0.55 \
    --gallery_dir dataset_preprocessed --probe_base .../probe_001
============================================================
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from palm_core import cfg as palm_cfg
from dorsal_core import cfg as dorsal_cfg
from palm_run import PalmVerifier
from dorsal_run import DorsalVerifier


# ============================================================
# VERIFIER MULTIMODALE
# ============================================================
class MultiModalVerifier:
    """Carica i due modelli gia' addestrati e fonde gli score a livello di similarita'."""

    def __init__(self, palm_checkpoint, dorsal_checkpoint, alpha: float = 0.5, device=None):
        self.palm = PalmVerifier(palm_checkpoint, device=device)
        self.dorsal = DorsalVerifier(dorsal_checkpoint, device=device)
        self.alpha = alpha   # peso del palmo; (1 - alpha) al dorso

    def embed(self, palm_base, dorsal_base):
        return self.palm.embed(palm_base), self.dorsal.embed(dorsal_base)

    @staticmethod
    def _cos(e1, e2):
        return F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()

    def fused_similarity(self, palm_base1, dorsal_base1, palm_base2, dorsal_base2):
        p1, d1 = self.embed(palm_base1, dorsal_base1)
        p2, d2 = self.embed(palm_base2, dorsal_base2)
        sim_palm = self._cos(p1, p2)
        sim_dorsal = self._cos(d1, d2)
        sim_fused = self.alpha * sim_palm + (1 - self.alpha) * sim_dorsal
        return sim_palm, sim_dorsal, sim_fused

    def verify(self, palm_base1, dorsal_base1, palm_base2, dorsal_base2, threshold: float):
        sim_palm, sim_dorsal, sim_fused = self.fused_similarity(
            palm_base1, dorsal_base1, palm_base2, dorsal_base2
        )
        return {
            "palm_similarity": sim_palm,
            "dorsal_similarity": sim_dorsal,
            "fused_similarity": sim_fused,
            "same_subject": bool(sim_fused >= threshold),
            "threshold": threshold,
            "alpha": self.alpha,
        }

    def identify(self, palm_probe, dorsal_probe, gallery: dict, threshold: float):
        """gallery: {subject_id: (palm_embedding, dorsal_embedding)}"""
        p_probe, d_probe = self.embed(palm_probe, dorsal_probe)
        best_subject, best_sim = None, -1.0
        per_subject_scores = {}

        for subject_id, (p_gal, d_gal) in gallery.items():
            sim_palm = self._cos(p_probe, p_gal)
            sim_dorsal = self._cos(d_probe, d_gal)
            sim_fused = self.alpha * sim_palm + (1 - self.alpha) * sim_dorsal
            per_subject_scores[subject_id] = {
                "palm_similarity": sim_palm, "dorsal_similarity": sim_dorsal,
                "fused_similarity": sim_fused,
            }
            if sim_fused > best_sim:
                best_subject, best_sim = subject_id, sim_fused

        identified = best_subject if best_sim >= threshold else None
        return {
            "identified_subject": identified,
            "best_similarity": best_sim,
            "threshold": threshold,
            "scores": per_subject_scores,   # utile per audit/debug
        }


# ============================================================
# RACCOLTA CAMPIONI (base path condiviso palmo/dorso per soggetto+acquisizione)
# ============================================================
def collect_paired_samples(data_dir, subject_ids=None):
    """
    Assume che ogni acquisizione abbia sia il record palmare che quello dorsale
    con lo stesso prefisso <subject>_<seq> nella stessa cartella soggetto
    (es. 0001_001_palm_hand.png e 0001_001_dorsal_hand.png), come prodotto
    da preProcessing.py sulle due mani/pose della stessa sessione.
    """
    data_dir = Path(data_dir)
    subject_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if subject_ids is not None:
        allowed = set(subject_ids)
        subject_dirs = [d for d in subject_dirs if d.name in allowed]

    samples = []
    for sdir in subject_dirs:
        palm_bases = {p.name.replace("_palm_hand.png", "") for p in sdir.glob("*_palm_hand.png")}
        dorsal_bases = {p.name.replace("_dorsal_hand.png", "") for p in sdir.glob("*_dorsal_hand.png")}
        for base in sorted(palm_bases & dorsal_bases):
            samples.append({
                "subject": sdir.name,
                "palm_base": str(sdir / base),
                "dorsal_base": str(sdir / base),
            })
    return samples


def build_verification_pairs(samples, max_pairs=20000, seed=42):
    rng = np.random.RandomState(seed)
    by_subject = {}
    for i, s in enumerate(samples):
        by_subject.setdefault(s["subject"], []).append(i)

    pos_pairs = []
    for idxs in by_subject.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pos_pairs.append((idxs[a], idxs[b]))
    rng.shuffle(pos_pairs)
    pos_pairs = pos_pairs[: max_pairs // 2]

    subjects = list(by_subject.keys())
    neg_pairs, attempts = [], 0
    while len(neg_pairs) < len(pos_pairs) and attempts < max_pairs * 20:
        s1, s2 = rng.choice(subjects, 2, replace=False)
        neg_pairs.append((rng.choice(by_subject[s1]), rng.choice(by_subject[s2])))
        attempts += 1
    return pos_pairs, neg_pairs


def compute_eer_from_scores(genuine, impostor):
    thresholds = np.linspace(-1, 1, 500)
    fars = np.array([(impostor >= t).mean() for t in thresholds])
    frrs = np.array([(genuine < t).mean() for t in thresholds])
    idx = np.argmin(np.abs(fars - frrs))
    return {"eer": float((fars[idx] + frrs[idx]) / 2), "eer_threshold": float(thresholds[idx])}


# ============================================================
# CALIBRAZIONE ALPHA + METRICHE
# ============================================================
def cmd_calibrate_cv(args):
    """
    Calibrazione ONESTA di alpha: usa i 5 checkpoint outer del nested_cv
    (palmo e dorso), valutando ogni fold SOLO sui suoi outer_test_subjects
    con il modello che non li ha mai visti in training. Aggrega le coppie
    genuine/impostor di tutti i fold prima di scegliere alpha.

    Richiede che palm_run.py nested_cv e dorsal_run.py nested_cv siano
    gia' stati eseguiti (stessi outer_folds, stesso seed, cosi' i fold
    palmo/dorso condividono gli stessi soggetti di test).
    """
    palm_summary = json.load(open(Path(args.palm_nested_dir) / "nested_cv_summary.json"))
    dorsal_summary = json.load(open(Path(args.dorsal_nested_dir) / "nested_cv_summary.json"))

    all_genuine = {a: [] for a in np.linspace(0.0, 1.0, 21)}
    all_impostor = {a: [] for a in np.linspace(0.0, 1.0, 21)}
    fold_reports = []

    for p_fold, d_fold in zip(palm_summary["outer_folds"], dorsal_summary["outer_folds"]):
        # i soggetti di test del fold sono salvati dentro il checkpoint stesso:
        p_ckpt = torch.load(p_fold["checkpoint"], map_location="cpu")
        d_ckpt = torch.load(d_fold["checkpoint"], map_location="cpu")
        test_subjects = p_ckpt["outer_test_subjects"]
        assert set(test_subjects) == set(d_ckpt["outer_test_subjects"]), \
            "I fold palmo/dorso non condividono gli stessi soggetti di test: usa lo stesso seed/outer_folds"

        verifier = MultiModalVerifier(p_fold["checkpoint"], d_fold["checkpoint"], alpha=0.5)
        samples = collect_paired_samples(args.data_dir, subject_ids=test_subjects)
        print(f"[Fold {p_fold['outer_fold']}] soggetti test={len(test_subjects)} campioni={len(samples)}")

        embeddings = [verifier.embed(s["palm_base"], s["dorsal_base"]) for s in samples]
        pos_pairs, neg_pairs = build_verification_pairs(samples, seed=100 + p_fold["outer_fold"])

        for alpha in all_genuine:
            def sim(i, j):
                p1, d1 = embeddings[i]; p2, d2 = embeddings[j]
                sp = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)).item()
                sd = F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item()
                return alpha * sp + (1 - alpha) * sd
            all_genuine[alpha].extend(sim(i, j) for i, j in pos_pairs)
            all_impostor[alpha].extend(sim(i, j) for i, j in neg_pairs)

        fold_reports.append({"outer_fold": p_fold["outer_fold"], "n_test_subjects": len(test_subjects),
                              "n_samples": len(samples)})
        del verifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grid_results, best_alpha, best_eer = [], None, float("inf")
    for alpha, genuine in all_genuine.items():
        res = compute_eer_from_scores(np.array(genuine), np.array(all_impostor[alpha]))
        grid_results.append({"alpha": float(alpha), **res})
        if res["eer"] < best_eer:
            best_eer, best_alpha = res["eer"], float(alpha)

    summary = {
        "protocol": "aggregated nested-cv fusion calibration (subject-disjoint per fold)",
        "fold_reports": fold_reports,
        "best_alpha": best_alpha,
        "fused_best_eer": best_eer,
        "alpha_grid": grid_results,
    }
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fusion_calibration_cv.json"
    json.dump(summary, open(out_path, "w", encoding="utf-8"), indent=2)

    print(f"\nEER fuso aggregato (onesto): {best_eer*100:.2f}% @ alpha={best_alpha:.2f}")
    print(f"Salvato in: {out_path}")
    print("\n>>> Usa questo alpha per addestrare il modello di produzione finale su TUTTI i soggetti,")
    print(">>> e per i comandi 'verify'/'identify' con quel modello.")


def cmd_calibrate(args):
    verifier = MultiModalVerifier(args.palm_checkpoint, args.dorsal_checkpoint, alpha=0.5)
    samples = collect_paired_samples(args.data_dir, subject_ids=None)
    print(f"Campioni palmo+dorso accoppiati: {len(samples)}")

    # Embedding di ogni campione, una sola volta.
    embeddings = []
    for s in samples:
        p_emb, d_emb = verifier.embed(s["palm_base"], s["dorsal_base"])
        embeddings.append((p_emb, d_emb))

    pos_pairs, neg_pairs = build_verification_pairs(samples)
    print(f"Coppie genuine: {len(pos_pairs)} | Coppie impostor: {len(neg_pairs)}")

    def sims_for_alpha(alpha):
        def sim(i, j):
            p1, d1 = embeddings[i]
            p2, d2 = embeddings[j]
            sp = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)).item()
            sd = F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item()
            return sp, sd, alpha * sp + (1 - alpha) * sd
        return sim

    # Grid search di alpha in [0, 1] che minimizza l'EER fuso.
    alpha_grid = np.linspace(0.0, 1.0, 21)
    best_alpha, best_eer = None, float("inf")
    grid_results = []

    for alpha in alpha_grid:
        sim_fn = sims_for_alpha(alpha)
        genuine = np.array([sim_fn(i, j)[2] for i, j in pos_pairs])
        impostor = np.array([sim_fn(i, j)[2] for i, j in neg_pairs])
        res = compute_eer_from_scores(genuine, impostor)
        grid_results.append({"alpha": float(alpha), **res})
        if res["eer"] < best_eer:
            best_eer, best_alpha = res["eer"], float(alpha)

    # Metriche per singolo stream (alpha=1 -> solo palmo, alpha=0 -> solo dorso).
    sim_palm_only = sims_for_alpha(1.0)
    sim_dorsal_only = sims_for_alpha(0.0)
    palm_eer = compute_eer_from_scores(
        np.array([sim_palm_only(i, j)[2] for i, j in pos_pairs]),
        np.array([sim_palm_only(i, j)[2] for i, j in neg_pairs]),
    )
    dorsal_eer = compute_eer_from_scores(
        np.array([sim_dorsal_only(i, j)[2] for i, j in pos_pairs]),
        np.array([sim_dorsal_only(i, j)[2] for i, j in neg_pairs]),
    )

    summary = {
        "n_samples": len(samples),
        "n_genuine_pairs": len(pos_pairs),
        "n_impostor_pairs": len(neg_pairs),
        "palm_only": palm_eer,
        "dorsal_only": dorsal_eer,
        "best_alpha": best_alpha,
        "fused_best_eer": best_eer,
        "alpha_grid": grid_results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fusion_calibration.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"EER solo palmo:  {palm_eer['eer']*100:.2f}%")
    print(f"EER solo dorso:  {dorsal_eer['eer']*100:.2f}%")
    print(f"EER fuso (best): {best_eer*100:.2f}%  @ alpha={best_alpha:.2f}")
    print(f"Metriche salvate in: {out_path}")


# ============================================================
# VERIFICA / IDENTIFICAZIONE SU NUOVI CAMPIONI
# ============================================================
def cmd_verify(args):
    verifier = MultiModalVerifier(args.palm_checkpoint, args.dorsal_checkpoint, alpha=args.alpha)
    result = verifier.verify(
        args.palm_base1, args.dorsal_base1 or args.palm_base1,
        args.palm_base2, args.dorsal_base2 or args.palm_base2,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))


def cmd_identify(args):
    verifier = MultiModalVerifier(args.palm_checkpoint, args.dorsal_checkpoint, alpha=args.alpha)
    samples = collect_paired_samples(args.gallery_dir)
    gallery = {}
    for s in samples:
        gallery[s["subject"]] = verifier.embed(s["palm_base"], s["dorsal_base"])

    result = verifier.identify(
        args.probe_palm_base, args.probe_dorsal_base or args.probe_palm_base,
        gallery, threshold=args.threshold,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "scores"}, indent=2))


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Fusione multimodale palmo+dorso")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser(
        "calibrate",
        help="calibra alpha su UN modello finale (ATTENZIONE: ottimistico se il modello ha visto tutti i soggetti)",
    )
    p_cal.add_argument("--palm_checkpoint", required=True)
    p_cal.add_argument("--dorsal_checkpoint", required=True)
    p_cal.add_argument("--data_dir", required=True)
    p_cal.add_argument("--out_dir", default="metrics_fusion")
    p_cal.set_defaults(func=cmd_calibrate)

    p_cal_cv = sub.add_parser(
        "calibrate_cv",
        help="calibra alpha in modo onesto usando i 5 checkpoint outer del nested_cv (subject-disjoint)",
    )
    p_cal_cv.add_argument("--palm_nested_dir", required=True,
                           help="es. models_final/nested_cv")
    p_cal_cv.add_argument("--dorsal_nested_dir", required=True,
                           help="es. models_final_dorsal/nested_cv")
    p_cal_cv.add_argument("--data_dir", required=True)
    p_cal_cv.add_argument("--out_dir", default="metrics_fusion")
    p_cal_cv.set_defaults(func=cmd_calibrate_cv)

    p_ver = sub.add_parser("verify", help="verifica 1:1 multimodale")
    p_ver.add_argument("--palm_checkpoint", required=True)
    p_ver.add_argument("--dorsal_checkpoint", required=True)
    p_ver.add_argument("--alpha", type=float, default=0.5)
    p_ver.add_argument("--palm_base1", required=True)
    p_ver.add_argument("--dorsal_base1", default=None)
    p_ver.add_argument("--palm_base2", required=True)
    p_ver.add_argument("--dorsal_base2", default=None)
    p_ver.add_argument("--threshold", type=float, default=0.55)
    p_ver.set_defaults(func=cmd_verify)

    p_id = sub.add_parser("identify", help="identificazione 1:N su gallery")
    p_id.add_argument("--palm_checkpoint", required=True)
    p_id.add_argument("--dorsal_checkpoint", required=True)
    p_id.add_argument("--alpha", type=float, default=0.5)
    p_id.add_argument("--gallery_dir", required=True)
    p_id.add_argument("--probe_palm_base", required=True)
    p_id.add_argument("--probe_dorsal_base", default=None)
    p_id.add_argument("--threshold", type=float, default=0.55)
    p_id.set_defaults(func=cmd_identify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()