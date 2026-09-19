"""
MULTIMODAL_FUSION - Valutazione multimodale su nuovo dataset
==============================================================

Adattato al repository:
https://github.com/patriziorenelli/AdvancedHandsBasedRecognition

Dataset atteso:
    Participants/
        P001/RGB/
            P001_L_dorsal_processed.png
            P001_L_palmar_processed.png
            P001_R_dorsal_processed.png
            P001_R_palmar_processed.png
        P002/RGB/
            ...

Il programma:
1. legge il nuovo dataset;
2. riapplica il preprocessing del progetto (MediaPipe + allineamento +
   ROI palmo + ROI nocche + preprocessing fotometrico);
3. usa PalmVerifier e DorsalVerifier dei modelli già addestrati;
4. calcola embedding una sola volta per campione;
5. esegue verifica 1:1 per:
      - PALMO
      - DORSO
      - FUSIONE PALMO+DORSO
6. calcola:
      EER, soglia EER, accuracy, balanced accuracy, FAR, FRR,
      ROC-AUC, TAR@FAR=1%, 5%, 10%;
7. esegue identificazione 1:N con:
      Rank-1, Rank-5, Rank-10, MRR;
8. salva tutto in:
      <out_dir>/
          results/
              summary.json
              verification_metrics.csv
              verification_pairs.csv
              identification_metrics.csv
              identification_details.csv
              alpha_calibration.json
          preprocessing_report.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from palm_run import PalmVerifier
from dorsal_run import DorsalVerifier


# ============================================================
# DATASET DISCOVERY
# ============================================================

IMAGE_RE = re.compile(
    r"^(?P<subject>P\d+)_"
    r"(?P<side>L|R)_"
    r"(?P<modality>palmar|dorsal)_"
    r"processed\.(?P<ext>png|jpg|jpeg|bmp)$",
    re.IGNORECASE,
)


def discover_raw_samples(data_dir: str):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset non trovato: {root}")

    records = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        m = IMAGE_RE.match(path.name)
        if not m:
            continue
        rec = {
            "subject": m.group("subject").upper(),
            "side": m.group("side").upper(),
            "modality": m.group("modality").lower(),
            "path": str(path),
        }
        records.append(rec)

    records.sort(key=lambda x: (x["subject"], x["side"], x["modality"]))

    if not records:
        raise RuntimeError(
            f"Nessuna immagine trovata in {root}. "
            "Attesi file PXXX_L_palmar_processed.png, "
            "PXXX_L_dorsal_processed.png, ecc."
        )

    return records


def build_raw_pairs(records):
    by_key = {}
    for r in records:
        by_key.setdefault((r["subject"], r["side"]), {})[r["modality"]] = r["path"]

    pairs = []
    missing = []

    for (subject, side), modalities in sorted(by_key.items()):
        if "palmar" not in modalities or "dorsal" not in modalities:
            missing.append({
                "subject": subject,
                "side": side,
                "available": sorted(modalities.keys()),
            })
            continue

        pairs.append({
            "subject": subject,
            "side": side,
            "palm_raw": modalities["palmar"],
            "dorsal_raw": modalities["dorsal"],
        })

    return pairs, missing


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_new_dataset(records, output_dir: Path):
    try:
        from preProcessing import init_worker, process_single_image
    except ImportError as exc:
        raise ImportError(
            "Impossibile importare preProcessing.py. "
            "Assicurati che multimodal_fusion.py sia nella root del repository "
            "e che MediaPipe sia installato."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    init_worker()

    report = {
        "n_input_images": len(records),
        "success": 0,
        "skipped": 0,
        "errors": 0,
        "results": [],
    }

    for rec in records:
        hand_side = "left" if rec["side"] == "L" else "right"
        is_dorsal = (rec["modality"] == "dorsal")

        task = (
            rec["path"],
            rec["subject"],
            hand_side,
            is_dorsal,
            1,
            str(output_dir),
        )

        result = process_single_image(task)
        result["subject"] = rec["subject"]
        result["side"] = rec["side"]
        result["modality"] = rec["modality"]
        report["results"].append(result)

        if result["status"] == "success":
            report["success"] += 1
        elif result["status"] == "skipped":
            report["skipped"] += 1
        else:
            report["errors"] += 1

        print(
            f"[PREPROCESS] {rec['subject']} {rec['side']} "
            f"{rec['modality']}: {result['status']}"
        )

    with open(output_dir.parent / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


# ============================================================
# PROCESSED DATASET DISCOVERY
# ============================================================

def find_processed_base(preprocessed_dir: Path, subject: str, side: str):
    subject_dir = preprocessed_dir / subject

    def find_unique_base(view_word: str, anchor_suffix: str):
        pattern = f"{subject}_{side}_{view_word}_*_{anchor_suffix}.png"
        matches = sorted(subject_dir.glob(pattern))

        if not matches:
            raise FileNotFoundError(
                f"Nessun file trovato per {subject} {side} ({view_word}) "
                f"con pattern '{pattern}' in {subject_dir}"
            )

        if len(matches) > 1:
            raise FileNotFoundError(
                f"Trovate {len(matches)} corrispondenze ambigue per "
                f"{subject} {side} ({view_word}) con pattern '{pattern}': "
                f"{[m.name for m in matches]}. Atteso un solo file."
            )

        match = matches[0]
        base_name = match.name[: -len(f"_{anchor_suffix}.png")]
        return subject_dir / base_name

    palm_base = find_unique_base("palmar", "palm_hand")
    dorsal_base = find_unique_base("dorsal", "dorsal_hand")

    palm_roi = Path(f"{palm_base}_palm_roi.png")

    if not palm_roi.exists():
        raise FileNotFoundError(
            f"Output palmo mancante per {subject} {side}: {palm_roi}"
        )

    return {
        "subject": subject,
        "side": side,
        "palm_base": str(palm_base),
        "dorsal_base": str(dorsal_base),
    }


def build_processed_pairs(preprocessed_dir: Path, raw_pairs):
    pairs = []
    failures = []

    for p in raw_pairs:
        try:
            pairs.append(
                find_processed_base(
                    preprocessed_dir,
                    p["subject"],
                    p["side"],
                )
            )
        except Exception as exc:
            failures.append({
                "subject": p["subject"],
                "side": p["side"],
                "error": str(exc),
            })

    return pairs, failures


# ============================================================
# EMBEDDING
# ============================================================

def compute_embeddings(pairs, palm_ckpt, dorsal_ckpt, device=None):
    print("\nCaricamento PalmVerifier...")
    palm = PalmVerifier(palm_ckpt, device=device)

    print("\nCaricamento DorsalVerifier...")
    dorsal = DorsalVerifier(dorsal_ckpt, device=device)

    embeddings = []

    for idx, p in enumerate(pairs, start=1):
        print(
            f"[EMBED] {idx}/{len(pairs)} "
            f"{p['subject']} {p['side']}"
        )

        p_emb = palm.embed(p["palm_base"])
        d_emb = dorsal.embed(p["dorsal_base"])

        embeddings.append({
            **p,
            "palm_embedding": p_emb,
            "dorsal_embedding": d_emb,
        })

    return embeddings


def cosine(e1, e2):
    return float(
        F.cosine_similarity(
            e1.unsqueeze(0),
            e2.unsqueeze(0),
        ).item()
    )


# ============================================================
# VERIFICATION PAIRS
# ============================================================

def make_verification_pairs(embeddings, max_impostor_pairs=None, seed=42):
    rng = random.Random(seed)

    by_subject = {}
    for i, x in enumerate(embeddings):
        by_subject.setdefault(x["subject"], {})[x["side"]] = i

    genuine = []
    for subject, sides in sorted(by_subject.items()):
        if "L" in sides and "R" in sides:
            genuine.append((sides["L"], sides["R"]))

    if not genuine:
        raise RuntimeError("Nessuna coppia genuine L-R disponibile.")

    all_impostor = []
    subjects = sorted(by_subject.keys())

    for a_idx in range(len(subjects)):
        s1 = subjects[a_idx]
        for b_idx in range(a_idx + 1, len(subjects)):
            s2 = subjects[b_idx]

            for i in by_subject[s1].values():
                for j in by_subject[s2].values():
                    all_impostor.append((i, j))

    rng.shuffle(all_impostor)

    n_imp = len(genuine)
    if max_impostor_pairs is not None:
        n_imp = min(n_imp, int(max_impostor_pairs))

    impostor = all_impostor[:n_imp]

    if len(genuine) > len(impostor):
        genuine = genuine[:len(impostor)]

    return genuine, impostor


# ============================================================
# METRICHE VERIFICATION
# ============================================================

def roc_auc(scores_genuine, scores_impostor):
    pos = np.asarray(scores_genuine, dtype=np.float64)
    neg = np.asarray(scores_impostor, dtype=np.float64)

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    scores = np.concatenate([pos, neg])
    labels = np.concatenate([
        np.ones(len(pos), dtype=np.int8),
        np.zeros(len(neg), dtype=np.int8),
    ])

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    ranks = np.empty(len(scores), dtype=np.float64)

    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    pos_ranks = ranks[labels == 1]
    n_pos = len(pos)
    n_neg = len(neg)

    u = pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def rates_at_threshold(genuine, impostor, threshold):
    genuine = np.asarray(genuine)
    impostor = np.asarray(impostor)

    far = float(np.mean(impostor >= threshold))
    frr = float(np.mean(genuine < threshold))

    tp = float(np.sum(genuine >= threshold))
    fn = float(np.sum(genuine < threshold))
    tn = float(np.sum(impostor < threshold))
    fp = float(np.sum(impostor >= threshold))

    total = tp + fn + tn + fp

    accuracy = (tp + tn) / total if total else float("nan")

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (tpr + tnr) / 2.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * tpr / (precision + tpr)
        if (precision + tpr)
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "far": far,
        "frr": frr,
        "tpr": tpr,
        "tnr": tnr,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def eer_metrics(genuine, impostor):
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)

    scores = np.unique(np.concatenate([genuine, impostor]))
    if len(scores) == 1:
        thresholds = scores
    else:
        thresholds = np.concatenate([
            [scores[0] - 1e-7],
            (scores[:-1] + scores[1:]) / 2.0,
            [scores[-1] + 1e-7],
        ])

    fars = np.array([np.mean(impostor >= t) for t in thresholds])
    frrs = np.array([np.mean(genuine < t) for t in thresholds])

    idx = int(np.argmin(np.abs(fars - frrs)))
    threshold = float(thresholds[idx])
    eer = float((fars[idx] + frrs[idx]) / 2.0)

    return {
        "eer": eer,
        "eer_threshold": threshold,
        "eer_far": float(fars[idx]),
        "eer_frr": float(frrs[idx]),
    }


def tar_at_far(genuine, impostor, target_far):
    impostor = np.asarray(impostor, dtype=np.float64)
    genuine = np.asarray(genuine, dtype=np.float64)

    if len(impostor) == 0:
        return {
            "target_far": target_far,
            "threshold": float("nan"),
            "actual_far": float("nan"),
            "tar": float("nan"),
        }

    candidates = np.unique(impostor)
    candidates = np.concatenate([candidates, [np.max(impostor) + 1e-8]])

    valid = []
    for t in candidates:
        far = float(np.mean(impostor >= t))
        if far <= target_far:
            tar = float(np.mean(genuine >= t))
            valid.append((tar, t, far))

    if not valid:
        t = float(np.max(candidates))
        return {
            "target_far": target_far,
            "threshold": t,
            "actual_far": float(np.mean(impostor >= t)),
            "tar": float(np.mean(genuine >= t)),
        }

    tar, threshold, actual_far = max(valid, key=lambda x: x[0])

    return {
        "target_far": target_far,
        "threshold": float(threshold),
        "actual_far": float(actual_far),
        "tar": float(tar),
    }


def evaluate_verification(genuine, impostor, fixed_threshold, name):
    eer = eer_metrics(genuine, impostor)
    fixed = rates_at_threshold(genuine, impostor, fixed_threshold)
    at_eer = rates_at_threshold(genuine, impostor, eer["eer_threshold"])

    return {
        "system": name,
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "eer": eer["eer"],
        "eer_percent": eer["eer"] * 100.0,
        "eer_threshold": eer["eer_threshold"],
        "roc_auc": roc_auc(genuine, impostor),

        "fixed_threshold": fixed,
        "eer_threshold_metrics": at_eer,

        "tar_at_far_1pct": tar_at_far(genuine, impostor, 0.01),
        "tar_at_far_5pct": tar_at_far(genuine, impostor, 0.05),
        "tar_at_far_10pct": tar_at_far(genuine, impostor, 0.10),
    }


# ============================================================
# SCORE EXTRACTION
# ============================================================

def compute_score_stats(embeddings, seed=42):
    genuine, impostor = make_verification_pairs(embeddings, seed=seed)
    scores = pair_scores(embeddings, genuine + impostor, alpha=0.5)

    def stats(arr):
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std < 1e-8:
            std = 1.0
        return mean, std

    palm_mean, palm_std = stats(scores["palm"])
    dorsal_mean, dorsal_std = stats(scores["dorsal"])

    return {
        "palm": {"mean": palm_mean, "std": palm_std},
        "dorsal": {"mean": dorsal_mean, "std": dorsal_std},
    }


def _normalize(value, stat):
    return (value - stat["mean"]) / stat["std"]


def pair_scores(embeddings, pairs, alpha, norm_stats=None):
    palm = []
    dorsal = []
    fused = []

    for i, j in pairs:
        sp = cosine(
            embeddings[i]["palm_embedding"],
            embeddings[j]["palm_embedding"],
        )
        sd = cosine(
            embeddings[i]["dorsal_embedding"],
            embeddings[j]["dorsal_embedding"],
        )

        if norm_stats is not None:
            sp_fusion = _normalize(sp, norm_stats["palm"])
            sd_fusion = _normalize(sd, norm_stats["dorsal"])
        else:
            sp_fusion = sp
            sd_fusion = sd

        sf = alpha * sp_fusion + (1.0 - alpha) * sd_fusion

        palm.append(sp)
        dorsal.append(sd)
        fused.append(sf)

    return {
        "palm": np.asarray(palm, dtype=np.float64),
        "dorsal": np.asarray(dorsal, dtype=np.float64),
        "fused": np.asarray(fused, dtype=np.float64),
    }


# ============================================================
# ALPHA CALIBRATION
# ============================================================

def calibrate_alpha(embeddings, subjects, seed=42, use_score_norm=False):
    subjects = sorted(set(subjects))

    if len(subjects) < 5:
        raise RuntimeError(
            "Servono almeno 5 soggetti per la calibrazione subject-disjoint."
        )

    rng = random.Random(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)

    n_cal = max(2, int(round(len(shuffled) * 0.20)))
    n_cal = min(n_cal, len(shuffled) - 2)

    calibration_subjects = set(shuffled[:n_cal])
    test_subjects = set(shuffled[n_cal:])

    cal_emb = [x for x in embeddings if x["subject"] in calibration_subjects]
    test_emb = [x for x in embeddings if x["subject"] in test_subjects]

    genuine, impostor = make_verification_pairs(cal_emb, seed=seed + 1)
    norm_stats = compute_score_stats(cal_emb, seed=seed) if use_score_norm else None

    grid = np.linspace(0.0, 1.0, 21)

    results = []
    best_alpha = None
    best_eer = float("inf")

    for alpha in grid:
        scores = pair_scores(
            cal_emb, genuine + impostor, float(alpha), norm_stats=norm_stats
        )

        n_g = len(genuine)
        g = scores["fused"][:n_g]
        imp = scores["fused"][n_g:]

        eer = eer_metrics(g, imp)["eer"]

        results.append({
            "alpha": float(alpha),
            "eer": float(eer),
            "eer_percent": float(eer * 100.0),
        })

        if eer < best_eer:
            best_eer = eer
            best_alpha = float(alpha)

    return {
        "alpha": best_alpha,
        "calibration_eer": best_eer,
        "calibration_eer_percent": best_eer * 100.0,
        "calibration_subjects": sorted(calibration_subjects),
        "test_subjects": sorted(test_subjects),
        "n_calibration_subjects": len(calibration_subjects),
        "n_test_subjects": len(test_subjects),
        "grid": results,
        "test_embeddings": test_emb,
        "norm_stats": norm_stats,
    }


# ============================================================
# IDENTIFICATION 1:N
# ============================================================

def identification_once(embeddings, gallery_side, probe_side, alpha, name, norm_stats=None):
    gallery = {
        x["subject"]: x
        for x in embeddings
        if x["side"] == gallery_side
    }

    probes = [
        x for x in embeddings
        if x["side"] == probe_side and x["subject"] in gallery
    ]

    if not gallery or not probes:
        raise RuntimeError(f"Identificazione {name}: gallery/probe vuote.")

    details = []
    rank1 = 0
    rank5 = 0
    rank10 = 0
    reciprocal_sum = 0.0

    for probe in probes:
        scores = []

        for subject, gal in gallery.items():
            sp = cosine(probe["palm_embedding"], gal["palm_embedding"])
            sd = cosine(probe["dorsal_embedding"], gal["dorsal_embedding"])

            if norm_stats is not None:
                sp_fusion = _normalize(sp, norm_stats["palm"])
                sd_fusion = _normalize(sd, norm_stats["dorsal"])
            else:
                sp_fusion = sp
                sd_fusion = sd

            sf = alpha * sp_fusion + (1.0 - alpha) * sd_fusion

            scores.append({
                "subject": subject,
                "palm_similarity": sp,
                "dorsal_similarity": sd,
                "fused_similarity": sf,
            })

        scores.sort(key=lambda x: x["fused_similarity"], reverse=True)

        ranked_subjects = [x["subject"] for x in scores]
        true_subject = probe["subject"]

        rank = ranked_subjects.index(true_subject) + 1

        rank1 += int(rank <= 1)
        rank5 += int(rank <= 5)
        rank10 += int(rank <= 10)
        reciprocal_sum += 1.0 / rank

        details.append({
            "system": name,
            "probe_subject": true_subject,
            "probe_side": probe_side,
            "gallery_side": gallery_side,
            "rank": rank,
            "predicted_subject": ranked_subjects[0],
            "top1_correct": bool(rank == 1),
            "top5_correct": bool(rank <= 5),
            "top10_correct": bool(rank <= 10),
            "top1_palm": max(scores, key=lambda x: x["palm_similarity"])["subject"],
            "top1_dorsal": max(scores, key=lambda x: x["dorsal_similarity"])["subject"],
        })

    n = len(probes)

    return {
        "system": name,
        "gallery_side": gallery_side,
        "probe_side": probe_side,
        "n_gallery_subjects": len(gallery),
        "n_probes": n,
        "rank1": rank1 / n,
        "rank1_percent": 100.0 * rank1 / n,
        "rank5": rank5 / n,
        "rank5_percent": 100.0 * rank5 / n,
        "rank10": rank10 / n,
        "rank10_percent": 100.0 * rank10 / n,
        "mrr": reciprocal_sum / n,
    }, details


def identification_all_systems(embeddings, alpha, norm_stats=None):
    all_metrics = []
    all_details = []

    for gallery_side, probe_side in [("L", "R"), ("R", "L")]:
        for system, a in [
            ("palm", 1.0),
            ("dorsal", 0.0),
            ("fused", alpha),
        ]:
            stats_for_call = norm_stats if system == "fused" else None

            metrics, details = identification_once(
                embeddings,
                gallery_side,
                probe_side,
                a,
                system,
                norm_stats=stats_for_call,
            )
            all_metrics.append(metrics)
            all_details.extend(details)

    for system in ["palm", "dorsal", "fused"]:
        rows = [x for x in all_metrics if x["system"] == system]
        if len(rows) == 2:
            all_metrics.append({
                "system": system,
                "gallery_side": "L/R mean",
                "probe_side": "R/L mean",
                "n_gallery_subjects": min(r["n_gallery_subjects"] for r in rows),
                "n_probes": sum(r["n_probes"] for r in rows),
                "rank1": float(np.mean([r["rank1"] for r in rows])),
                "rank1_percent": float(np.mean([r["rank1_percent"] for r in rows])),
                "rank5": float(np.mean([r["rank5"] for r in rows])),
                "rank5_percent": float(np.mean([r["rank5_percent"] for r in rows])),
                "rank10": float(np.mean([r["rank10"] for r in rows])),
                "rank10_percent": float(np.mean([r["rank10_percent"] for r in rows])),
                "mrr": float(np.mean([r["mrr"] for r in rows])),
            })

    return all_metrics, all_details


# ============================================================
# OUTPUT
# ============================================================

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_verification_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "n_genuine",
                "n_impostor",
                "eer",
                "eer_percent",
                "eer_threshold",
                "roc_auc",
                "fixed_threshold",
                "fixed_accuracy",
                "fixed_balanced_accuracy",
                "fixed_far",
                "fixed_frr",
                "fixed_precision",
                "fixed_f1",
                "eer_accuracy",
                "eer_balanced_accuracy",
                "eer_far",
                "eer_frr",
                "eer_precision",
                "eer_f1",
                "tar_at_far_1pct",
                "tar_at_far_5pct",
                "tar_at_far_10pct",
            ],
        )
        writer.writeheader()

        for r in rows:
            fixed = r["fixed_threshold_metrics"]
            eerm = r["eer_threshold_metrics"]

            writer.writerow({
                "system": r["system"],
                "n_genuine": r["n_genuine"],
                "n_impostor": r["n_impostor"],
                "eer": r["eer"],
                "eer_percent": r["eer_percent"],
                "eer_threshold": r["eer_threshold"],
                "roc_auc": r["roc_auc"],
                "fixed_threshold": fixed["threshold"],
                "fixed_accuracy": fixed["accuracy"],
                "fixed_balanced_accuracy": fixed["balanced_accuracy"],
                "fixed_far": fixed["far"],
                "fixed_frr": fixed["frr"],
                "fixed_precision": fixed["precision"],
                "fixed_f1": fixed["f1"],
                "eer_accuracy": eerm["accuracy"],
                "eer_balanced_accuracy": eerm["balanced_accuracy"],
                "eer_far": eerm["far"],
                "eer_frr": eerm["frr"],
                "eer_precision": eerm["precision"],
                "eer_f1": eerm["f1"],
                "tar_at_far_1pct": r["tar_at_far_1pct"]["tar"],
                "tar_at_far_5pct": r["tar_at_far_5pct"]["tar"],
                "tar_at_far_10pct": r["tar_at_far_10pct"]["tar"],
            })


def save_identification_csv(path, rows):
    fields = [
        "system",
        "gallery_side",
        "probe_side",
        "n_gallery_subjects",
        "n_probes",
        "rank1",
        "rank1_percent",
        "rank5",
        "rank5_percent",
        "rank10",
        "rank10_percent",
        "mrr",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_details_csv(path, rows):
    fields = [
        "system",
        "probe_subject",
        "probe_side",
        "gallery_side",
        "rank",
        "predicted_subject",
        "top1_correct",
        "top5_correct",
        "top10_correct",
        "top1_palm",
        "top1_dorsal",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_pair_scores_csv(path, embeddings, genuine, impostor, alpha, norm_stats=None):
    rows = []

    for label, pairs in [("genuine", genuine), ("impostor", impostor)]:
        for i, j in pairs:
            sp = cosine(embeddings[i]["palm_embedding"], embeddings[j]["palm_embedding"])
            sd = cosine(embeddings[i]["dorsal_embedding"], embeddings[j]["dorsal_embedding"])

            if norm_stats is not None:
                sp_fusion = _normalize(sp, norm_stats["palm"])
                sd_fusion = _normalize(sd, norm_stats["dorsal"])
            else:
                sp_fusion = sp
                sd_fusion = sd

            sf = alpha * sp_fusion + (1.0 - alpha) * sd_fusion

            rows.append({
                "label": label,
                "subject1": embeddings[i]["subject"],
                "side1": embeddings[i]["side"],
                "subject2": embeddings[j]["subject"],
                "side2": embeddings[j]["side"],
                "palm_similarity": sp,
                "dorsal_similarity": sd,
                "fused_similarity": sf,
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "label",
            "subject1",
            "side1",
            "subject2",
            "side2",
            "palm_similarity",
            "dorsal_similarity",
            "fused_similarity",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_dataset(args):
    out_root = Path(args.out_dir)

    # Rilevamento automatico e flessibile della cartella preelaborata
    if (out_root / "preprocessed").exists():
        preprocessed_dir = out_root / "preprocessed"
    else:
        preprocessed_dir = out_root

    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. DISCOVERY
    records = discover_raw_samples(args.data_dir)
    raw_pairs, missing_pairs = build_raw_pairs(records)

    print("\n" + "=" * 72)
    print("NUOVO DATASET")
    print("=" * 72)
    print(f"Immagini trovate:        {len(records)}")
    print(f"Coppie multimodali:      {len(raw_pairs)}")
    print(f"Coppie incomplete:       {len(missing_pairs)}")
    print(f"Soggetti:                {len(set(x['subject'] for x in raw_pairs))}")

    # 2. PREPROCESSING
    if args.skip_preprocessing:
        print("\nPreprocessing saltato (--skip_preprocessing).")
    else:
        print("\n" + "=" * 72)
        print("PREPROCESSING")
        print("=" * 72)

        if preprocessed_dir.exists() and args.clean_preprocessed:
            shutil.rmtree(preprocessed_dir)

        preprocessing_report = preprocess_new_dataset(
            records,
            preprocessed_dir,
        )

        print(
            f"\nPreprocessing: success={preprocessing_report['success']} "
            f"skipped={preprocessing_report['skipped']} "
            f"errors={preprocessing_report['errors']}"
        )

    # 3. PROCESSED PAIRS
    processed_pairs, preprocess_failures = build_processed_pairs(
        preprocessed_dir,
        raw_pairs,
    )

    print(f"\nCampioni multimodali utilizzabili: {len(processed_pairs)}")

    if len(processed_pairs) < 4:
        raise RuntimeError("Troppi pochi campioni dopo il preprocessing.")

    # 4. EMBEDDINGS
    embeddings = compute_embeddings(
        processed_pairs,
        args.palm_checkpoint,
        args.dorsal_checkpoint,
        device=args.device,
    )

    # 5. ALPHA
    subjects = sorted(set(x["subject"] for x in embeddings))

    calibration_info = None
    norm_stats = None
    use_score_norm = (args.score_norm == "zscore")

    if args.calibrate_alpha:
        calibration_info = calibrate_alpha(
            embeddings,
            subjects,
            seed=args.seed,
            use_score_norm=use_score_norm,
        )

        alpha = calibration_info["alpha"]
        norm_stats = calibration_info["norm_stats"]

        save_json(
            results_dir / "alpha_calibration.json",
            {k: v for k, v in calibration_info.items() if k != "test_embeddings"},
        )

        evaluation_embeddings = calibration_info["test_embeddings"]

        print(f"\nAlpha calibrato subject-disjoint: {alpha:.2f}")
        print(f"Soggetti calibration: {calibration_info['n_calibration_subjects']}")
        print(f"Soggetti test:        {calibration_info['n_test_subjects']}")
    else:
        alpha = float(args.alpha)
        evaluation_embeddings = embeddings

        if use_score_norm:
            print(
                "\n[ATTENZIONE] --score_norm zscore senza --calibrate_alpha: "
                "le statistiche vengono stimate sullo stesso set di test."
            )
            norm_stats = compute_score_stats(evaluation_embeddings, seed=args.seed)

    # 6. VERIFICATION
    print("\n" + "=" * 72)
    print("VERIFICA 1:1")
    print("=" * 72)

    genuine, impostor = make_verification_pairs(
        evaluation_embeddings,
        max_impostor_pairs=args.max_impostor_pairs,
        seed=args.seed,
    )

    print(f"Coppie genuine:  {len(genuine)}")
    print(f"Coppie impostor: {len(impostor)}")
    print(f"Alpha:            {alpha:.2f}")

    scores = pair_scores(
        evaluation_embeddings,
        genuine + impostor,
        alpha,
        norm_stats=norm_stats,
    )

    n_g = len(genuine)

    systems = {
        "palm": (scores["palm"][:n_g], scores["palm"][n_g:]),
        "dorsal": (scores["dorsal"][:n_g], scores["dorsal"][n_g:]),
        "fused": (scores["fused"][:n_g], scores["fused"][n_g:]),
    }

    verification_results = []

    for name, (g, imp) in systems.items():
        metrics = evaluate_verification(
            g,
            imp,
            fixed_threshold=args.threshold,
            name=name,
        )
        metrics["fixed_threshold_metrics"] = metrics.pop("fixed_threshold")
        verification_results.append(metrics)

        print(
            f"{name:>8s}: "
            f"EER={metrics['eer_percent']:.2f}% | "
            f"AUC={metrics['roc_auc']:.4f} | "
            f"Acc@{args.threshold:.2f}="
            f"{metrics['fixed_threshold_metrics']['accuracy']*100:.2f}%"
        )

    save_verification_csv(results_dir / "verification_metrics.csv", verification_results)
    save_pair_scores_csv(
        results_dir / "verification_pairs.csv",
        evaluation_embeddings,
        genuine,
        impostor,
        alpha,
        norm_stats=norm_stats,
    )

    # 7. IDENTIFICATION
    print("\n" + "=" * 72)
    print("IDENTIFICAZIONE 1:N")
    print("=" * 72)

    identification_metrics, identification_details = identification_all_systems(
        evaluation_embeddings,
        alpha,
        norm_stats=norm_stats,
    )

    for r in identification_metrics:
        if r["gallery_side"] == "L/R mean":
            print(
                f"{r['system']:>8s}: "
                f"Rank-1={r['rank1_percent']:.2f}% | "
                f"Rank-5={r['rank5_percent']:.2f}% | "
                f"Rank-10={r['rank10_percent']:.2f}% | "
                f"MRR={r['mrr']:.4f}"
            )

    save_identification_csv(results_dir / "identification_metrics.csv", identification_metrics)
    save_details_csv(results_dir / "identification_details.csv", identification_details)

    # 8. SUMMARY JSON
    summary = {
        "protocol": {
            "dataset_type": "external_new_dataset",
            "input_format": "PXXX_L/R_palmar/dorsal_processed.png",
            "verification": "L-vs-R genuine; subject-disjoint impostors",
            "identification": "L->R and R->L",
            "alpha": alpha,
            "alpha_calibrated_subject_disjoint": bool(args.calibrate_alpha),
            "score_normalization": args.score_norm,
            "fixed_verification_threshold": args.threshold,
            "seed": args.seed,
        },
        "dataset": {
            "input_images": len(records),
            "multimodal_pairs_found": len(raw_pairs),
            "multimodal_pairs_usable": len(processed_pairs),
            "n_subjects": len(set(x["subject"] for x in embeddings)),
            "n_embedding_samples": len(embeddings),
            "missing_pairs": missing_pairs,
            "preprocess_failures": preprocess_failures,
        },
        "verification": verification_results,
        "identification": identification_metrics,
        "alpha_calibration": (
            {k: v for k, v in calibration_info.items() if k != "test_embeddings"}
            if calibration_info is not None
            else None
        ),
    }

    save_json(results_dir / "summary.json", summary)

    # 9. HUMAN-READABLE TXT
    with open(results_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("RISULTATI NUOVO DATASET - PALMO + DORSO\n")
        f.write("=" * 72 + "\n\n")

        f.write(f"Soggetti utilizzabili: {summary['dataset']['n_subjects']}\n")
        f.write(f"Campioni multimodali: {summary['dataset']['n_embedding_samples']}\n")
        f.write(f"Alpha fusione: {alpha:.2f}\n")
        f.write(f"Soglia verifica fissa: {args.threshold:.4f}\n\n")

        f.write("VERIFICATION 1:1\n")
        f.write("-" * 72 + "\n")

        for r in verification_results:
            fixed = r["fixed_threshold_metrics"]
            eer_m = r["eer_threshold_metrics"]

            f.write(
                f"\n{r['system'].upper()}\n"
                f"  EER:                    {r['eer']*100:.4f}%\n"
                f"  EER threshold:          {r['eer_threshold']:.6f}\n"
                f"  ROC-AUC:                {r['roc_auc']:.6f}\n"
                f"  Accuracy @ {args.threshold:.2f}: {fixed['accuracy']*100:.4f}%\n"
                f"  Balanced accuracy @ {args.threshold:.2f}: {fixed['balanced_accuracy']*100:.4f}%\n"
                f"  FAR @ {args.threshold:.2f}: {fixed['far']*100:.4f}%\n"
                f"  FRR @ {args.threshold:.2f}: {fixed['frr']*100:.4f}%\n"
                f"  Accuracy @ EER threshold: {eer_m['accuracy']*100:.4f}%\n"
                f"  TAR @ FAR 1%:           {r['tar_at_far_1pct']['tar']*100:.4f}%\n"
                f"  TAR @ FAR 5%:           {r['tar_at_far_5pct']['tar']*100:.4f}%\n"
                f"  TAR @ FAR 10%:          {r['tar_at_far_10pct']['tar']*100:.4f}%\n"
            )

        f.write("\nIDENTIFICATION 1:N\n")
        f.write("-" * 72 + "\n")

        for r in identification_metrics:
            if r["gallery_side"] == "L/R mean":
                f.write(
                    f"\n{r['system'].upper()}\n"
                    f"  Rank-1:  {r['rank1_percent']:.4f}%\n"
                    f"  Rank-5:  {r['rank5_percent']:.4f}%\n"
                    f"  Rank-10: {r['rank10_percent']:.4f}%\n"
                    f"  MRR:     {r['mrr']:.6f}\n"
                )

    print("\n" + "=" * 72)
    print("TEST COMPLETATO")
    print("=" * 72)
    print(f"Risultati: {results_dir}")
    print(f"Summary:   {results_dir / 'summary.json'}")
    print(f"CSV:       {results_dir / 'verification_metrics.csv'}")
    print(f"CSV:       {results_dir / 'identification_metrics.csv'}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test multimodale palmo+dorso sul nuovo dataset "
            "PXXX_L/R_palmar/dorsal_processed.png"
        )
    )

    parser.add_argument(
        "--data_dir",
        required=True,
        help="Cartella Participants del nuovo dataset, es. dataset_zenodo/Participants",
    )

    parser.add_argument(
        "--palm_checkpoint",
        default="models_final/palm_embedding_final.pt",
    )

    parser.add_argument(
        "--dorsal_checkpoint",
        default="models_final_dorsal/dorsal_embedding_final.pt",
    )

    parser.add_argument(
        "--out_dir",
        default="new_dataset_preprocessed",
        help="Cartella del dataset preelaborato (default: new_dataset_preprocessed)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.50,
        help="Peso palmo nella score fusion.",
    )

    parser.add_argument(
        "--score_norm",
        choices=["none", "zscore"],
        default="none",
        help="Standardizzazione z-score pre-fusione.",
    )

    parser.add_argument(
        "--calibrate_alpha",
        action="store_true",
        help="Calibra alpha su soggetti separati dal test.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Soglia fissa per Accuracy/FAR/FRR.",
    )

    parser.add_argument(
        "--max_impostor_pairs",
        type=int,
        default=None,
        help="Numero massimo di coppie impostor.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default=None,
        help="es. cuda oppure cpu.",
    )

    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Usa le immagini già preelaborate presenti in out_dir.",
    )

    parser.add_argument(
        "--clean_preprocessed",
        action="store_true",
        help="Cancella la cartella preprocessed prima del preprocessing.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha deve essere compreso tra 0 e 1.")

    evaluate_dataset(args)


if __name__ == "__main__":
    main()