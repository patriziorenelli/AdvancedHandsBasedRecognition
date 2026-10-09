"""
MULTIMODAL_FUSION v6 - piu' dataset, NO augmentation, NO leakage
=================================================================

Valuta palmo+dorso (verifica 1:1, identificazione closed-set 1:N, identificazione
open-set) usando UNO O PIU' dataset insieme, senza alcuna data augmentation.

Input: SOLO cartelle gia' preprocessate (nessun raw, nessun preprocessing), una per dataset:
  --dataset_pre TAG=DIR   cerca ricorsivamente *_palm_hand.png / *_palm_roi.png (+ crop nocche)
                          e *_dorsal_hand.png. Formati dei nomi riconosciuti:
                            0_dorsal left_017            (stile 11k Hands)
                            P001_1_L_palmar_processed    (stile PXXX)
  L'opzione si ripete, una volta per dataset (es. 11k e new).

Come vengono usati i due dataset
  - I soggetti sono prefissati col TAG ("11k:12", "new:P007"): nessuna collisione di ID.
  - Si eseguono valutazioni SEPARATE per ogni dataset e una valutazione POOLED.
  - In verifica gli impostor sono solo dello STESSO dataset, e in identificazione la
    gallery contiene solo identita' dello stesso dataset del probe: confrontare persone
    di dataset diversi e' banale (sensore/illuminazione diversi) e gonfierebbe i risultati.
  - DEV/TEST e split open-set sono stratificati per dataset.

Anti-leakage
  - --train_log: log JSONL di training (palmo e dorso) -> soggetti di train/eval da escludere.
  - --leak_check_tags: i TAG dei dataset da cui provengono i dati di training. Il confronto
    degli ID si fa SOLO per quei tag (ID di dataset diversi non sono confrontabili).
  - --dev_fraction: alpha e statistiche z-score si calibrano su soggetti DEV disgiunti dal TEST.

Nessuna augmentation: solo scatti reali; le mani con un solo scatto vengono escluse.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from palm_run import PalmVerifier
from dorsal_run import DorsalVerifier


# ============================================================
# DISCOVERY DIRETTA DAL PREPROCESSED (--preprocessed_only)
# ============================================================

_PRE_ANCHORS = {
    "_palm_hand.png": "palmar",
    "_dorsal_hand.png": "dorsal",
}


def _parse_preprocessed_name(path: Path, base_name: str):
    """
    Interpreta il nome base di un file preprocessato (nome senza il suffisso
    _palm_hand.png / _dorsal_hand.png). Formati supportati:
      A) P001_1_L_palmar_processed   (soggetto PXXX, scatto, lato L/R)
      B) 0_dorsal left_017           (11k Hands: id, vista + lato, indice immagine)
    Il soggetto e' il primo token (numerico o PXXX); se non lo e', si usa il
    nome della cartella padre. Lato: token L/R o left/right. Indice scatto:
    ultimo token numerico dopo il soggetto. Tag augN opzionale.
    Ritorna dict(subject, side, token, order, aug) oppure None.
    """
    tokens = [t for t in base_name.split("_") if t != ""]
    if not tokens:
        return None

    if re.fullmatch(r"P?\d+", tokens[0], flags=re.IGNORECASE):
        subject = tokens[0].upper()
    elif re.fullmatch(r"P?\d+", path.parent.name, flags=re.IGNORECASE):
        subject = path.parent.name.upper()
    else:
        return None

    words = [w for t in tokens[1:] for w in t.split()]
    side = None
    for w in words:
        wl = w.lower()
        if wl in ("l", "left"):
            side = "L"
            break
        if wl in ("r", "right"):
            side = "R"
            break
    if side is None:
        return None

    token = None
    for w in words:
        if re.fullmatch(r"s?\d+", w, flags=re.IGNORECASE):
            token = w.lower()
    aug = next((w.lower() for w in words if re.fullmatch(r"aug\d+", w, flags=re.IGNORECASE)), None)
    order = int(re.sub(r"\D", "", token)) if token else 0
    return {"subject": subject, "side": side, "token": token, "order": order, "aug": aug}


def discover_preprocessed_samples(preprocessed_dir: Path, dorsal_dir: Path | None = None):
    """
    Costruisce direttamente i campioni multimodali dalla cartella del
    preprocessing gia' calcolato, senza leggere il raw e senza augmentation.
    Cerca ricorsivamente *_palm_hand.png (palmo) e *_dorsal_hand.png (dorso).

    Accoppiamento palmo+dorso per ogni mano (soggetto, lato):
      - se palmo e dorso hanno lo STESSO insieme di indici di scatto, si
        accoppia per indice;
      - altrimenti (es. 11k Hands, dove palmo e dorso sono foto distinte con
        indici diversi) si accoppia per ORDINE: k-esimo palmo con k-esimo
        dorso, in ordine di indice. Gli scatti in eccesso di una modalita'
        vengono scartati.
    Ritorna (pairs, records, missing_pairs, failures, unparsed).
    """
    # Se dorsal_dir e' specificata, palmo e dorso possono risiedere in cartelle separate.
    palm_root = Path(preprocessed_dir)
    dorsal_root = Path(dorsal_dir) if dorsal_dir is not None else palm_root
    groups = {}
    unparsed = []
    seen = set()
    duplicates = []

    roots = [(palm_root, "palmar", ("_palm_hand.png", "_palm_roi.png")),
             (dorsal_root, "dorsal", ("_dorsal_hand.png",))]
    for scan_root, expected_modality, anchors in roots:
        if not scan_root.exists():
            raise FileNotFoundError(f"Cartella preprocessata non trovata: {scan_root}")
        for path in sorted(scan_root.rglob("*.png")):
            anchor = next((a for a in anchors if path.name.endswith(a)), None)
            if anchor is None or anchor == "_palm_roi.png":
                continue
            modality = expected_modality
            base_name = path.name[: -len(anchor)]
            info = _parse_preprocessed_name(path, base_name)
            if info is None:
                unparsed.append(str(path))
                continue
            info["base"] = str(path.with_name(base_name))
            info["path"] = str(path)
            dup_key = (info["subject"], info["side"], modality, info["token"], info["aug"])
            if info["token"] is not None and dup_key in seen:
                duplicates.append(info["base"])
                continue
            seen.add(dup_key)
            groups.setdefault((info["subject"], info["side"]), {"palmar": [], "dorsal": []})[modality].append(info)
    if duplicates:
        raise RuntimeError(
            f"{len(duplicates)} file preprocessati duplicati (stesso soggetto, lato, "
            f"vista e indice in cartelle diverse). Esempio: {duplicates[0]}"
        )

    if not groups:
        hint = ("\nEsempi di file non riconosciuti:\n  " + "\n  ".join(unparsed[:5])) if unparsed else ""
        raise RuntimeError(
            f"Nessun campione preprocessato utilizzabile in {palm_root} / {dorsal_root}. Attesi file "
            "*_palm_hand.png / *_dorsal_hand.png con soggetto (numero o PXXX) come "
            "primo token e lato L/R o left/right nel nome." + hint
        )

    def sort_key(x):
        return (x["order"], x["base"])

    pairs, records, missing, failures = [], [], [], []
    n_by_token = n_by_order = n_dropped = 0

    for (subject, side), mods in sorted(groups.items()):
        P = sorted(mods["palmar"], key=sort_key)
        D = sorted(mods["dorsal"], key=sort_key)
        if not P or not D:
            missing.append({"subject": subject, "side": side, "sample_id": "*",
                            "available": [m for m in ("palmar", "dorsal") if mods[m]]})
            continue

        ptok = [x["token"] for x in P]
        dtok = [x["token"] for x in D]
        by_token = (None not in ptok and set(ptok) == set(dtok)
                    and len(set(ptok)) == len(ptok) and len(set(dtok)) == len(dtok))
        if by_token:
            dmap = {x["token"]: x for x in D}
            matched = [(x, dmap[x["token"]]) for x in P]
            n_by_token += 1
        else:
            matched = list(zip(P, D))
            n_by_order += 1
            n_dropped += abs(len(P) - len(D))

        for k, (pi, di) in enumerate(matched, start=1):
            aug = pi["aug"] or di["aug"]
            base_id = pi["token"] if by_token else str(k)
            sample_id = f"{base_id}_{aug}" if aug else base_id
            palm_roi = Path(f"{pi['base']}_palm_roi.png")
            if not palm_roi.exists():
                failures.append({"subject": subject, "side": side, "sample_id": sample_id,
                                 "error": f"Output palmo mancante: {palm_roi}"})
                continue
            pairs.append({"subject": subject, "side": side, "sample_id": sample_id,
                          "palm_base": pi["base"], "dorsal_base": di["base"]})
            for modality, it in (("palmar", pi), ("dorsal", di)):
                records.append({"subject": subject, "side": side, "sample_id": sample_id,
                                "modality": modality, "is_augmented": bool(aug),
                                "path": it["path"]})

    print(f"[PAIRING] mani accoppiate per indice: {n_by_token} | per ordine: {n_by_order} "
          f"| scatti scartati per numero diverso palmo/dorso: {n_dropped}")
    if n_by_order:
        print("[PAIRING] nota: con accoppiamento per ordine palmo e dorso di uno stesso "
              "campione sono foto distinte della stessa mano.")

    return pairs, records, missing, failures, unparsed


# ============================================================
# ANTI-LEAKAGE (v5): esclusione soggetti di training + split dev/test
# ============================================================

def normalize_subject_id(s) -> str:
    """'P001' -> '1', '0012' -> '12', '0' -> '0': confronto robusto tra ID dei log e dei file."""
    s = re.sub(r"^[Pp]", "", str(s).strip())
    return s.lstrip("0") or "0"


def load_training_subjects(log_paths):
    """
    Legge uno o piu' log JSONL di training (evento 'dataset') e ritorna
    (train_ids, eval_ids, dettagli). Gli ID sono normalizzati. L'unione tra
    piu' log (es. palmo + dorso) e' la scelta sicura: un soggetto visto da
    QUALUNQUE modello e' considerato contaminato.
    """
    train_ids, eval_ids, details = set(), set(), []
    for p in log_paths:
        found = False
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("event") != "dataset":
                    continue
                found = True
                tr = {normalize_subject_id(x) for x in ev.get("train_subjects", [])}
                evs = {normalize_subject_id(x) for x in ev.get("eval_subjects", [])}
                train_ids |= tr
                eval_ids |= evs
                details.append({"log": str(p), "run_id": ev.get("run_id"),
                                "n_train_subjects": len(tr), "n_eval_subjects": len(evs)})
        if not found:
            raise RuntimeError(f"Nessun evento 'dataset' con i soggetti in {p}: log non valido?")
    return train_ids, eval_ids, details


def _label_dataset(label) -> str:
    """'11k:12' / '11k:12_L' -> '11k'. Stringa vuota se non c'e' il tag."""
    t = str(label)
    return t.split(":", 1)[0] if ":" in t else ""


def apply_leak_filter(pairs, train_ids, eval_ids, policy, check_tags):
    """
    Scarta i soggetti gia' visti in training, SOLO per i dataset in `check_tags`
    (quelli da cui provengono i dati di training). I dataset non in check_tags sono
    considerati disgiunti dal training: ID di dataset diversi non sono confrontabili.
      strict     -> scarta soggetti di TRAIN e di EVAL (l'eval ha guidato early stopping
                    e scelta del best checkpoint)
      train_only -> scarta solo i soggetti di TRAIN
    """
    banned = set(train_ids) | (set(eval_ids) if policy == "strict" else set())
    check = set(check_tags)

    def contaminated(p):
        return p["dataset"] in check and normalize_subject_id(p["raw_subject"]) in banned

    clean = [p for p in pairs if not contaminated(p)]
    per_dataset = {}
    for tag in sorted({p["dataset"] for p in pairs}):
        subs = {p["raw_subject"] for p in pairs if p["dataset"] == tag}
        checked = tag in check
        after = {p["raw_subject"] for p in clean if p["dataset"] == tag}
        per_dataset[tag] = {
            "checked_against_training": checked,
            "n_subjects_before": len(subs),
            "n_overlap_train": len([x for x in subs if checked and normalize_subject_id(x) in train_ids]),
            "n_overlap_eval": len([x for x in subs if checked and normalize_subject_id(x) in eval_ids]),
            "n_subjects_after": len(after),
            "removed_subjects": sorted(subs - after),
        }
    return clean, {"policy": policy, "per_dataset": per_dataset,
                   "n_pairs_before": len(pairs), "n_pairs_after": len(clean)}


def split_dev_test(embeddings, dev_fraction, seed):
    """
    Split subject-disjoint STRATIFICATO per dataset: DEV (solo per alpha e statistiche
    z-score) e TEST (solo per i numeri finali). Il test mantiene >= 4 soggetti per dataset.
    """
    by_ds = {}
    for x in embeddings:
        by_ds.setdefault(x.get("dataset", ""), set()).add(x["subject"])
    rng = random.Random(seed + 7)
    dev = set()
    for ds in sorted(by_ds):
        subs = sorted(by_ds[ds])
        rng.shuffle(subs)
        n_dev = min(max(int(round(dev_fraction * len(subs))), 0), max(len(subs) - 4, 0))
        dev |= set(subs[:n_dev])
    all_subs = {x["subject"] for x in embeddings}
    return ([x for x in embeddings if x["subject"] in dev],
            [x for x in embeddings if x["subject"] not in dev],
            sorted(dev), sorted(all_subs - dev))


# ============================================================
# EMBEDDING (invariato)
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
            f"{p['subject']} {p['side']} s{p['sample_id']}"
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
# VERIFICATION PAIRS - PROTOCOLLO CORRETTO, NIENTE FALLBACK SILENZIOSO
# ============================================================

def make_verification_pairs(embeddings, max_impostor_pairs=None, seed=42):
    """
    Ritorna (genuine, impostor, protocol_used).

    protocol_used e' sempre "same_hand_multi_sample": coppie genuine = stessa
    mano (stesso subject E stesso side), scatti (sample_id) diversi. Non esiste
    alcun fallback che mescoli mano sx e dx: se una mano non ha abbastanza
    scatti, va risolto a monte con l'augmentation automatica
    (ensure_min_samples_per_hand), non qui.

    NOTA (modifica): le coppie impostor NON vengono piu' troncate al numero
    di coppie genuine. EER/ROC-AUC/TAR@FAR sono tassi calcolati separatamente
    sulle due distribuzioni (genuine, impostor): non richiedono classi
    bilanciate, e usare TUTTE le coppie impostor disponibili da' stime molto
    piu' stabili, soprattutto per TAR a FAR bassi (es. 1%), dove con poche
    centinaia di impostor la soglia e' quantizzata e rumorosa. Se non vuoi
    usarle tutte per motivi di tempo di calcolo, passa esplicitamente
    --max_impostor_pairs. Le coppie genuine non vengono piu' troncate in
    funzione del numero di impostor: restano tutte quelle disponibili nei
    dati (limite strutturale: una coppia per ogni mano con esattamente 2
    scatti).
    """
    rng = random.Random(seed)

    genuine = []
    all_impostor = []
    n_samples = len(embeddings)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            e1, e2 = embeddings[i], embeddings[j]
            if e1["subject"] == e2["subject"] and e1["side"] == e2["side"]:
                genuine.append((i, j))

    protocol_used = "same_hand_multi_sample"

    if not genuine:
        raise RuntimeError(
            "Nessuna coppia 'stessa mano, scatti diversi' disponibile: servono "
            "almeno 2 scatti REALI per mano (nessuna augmentation in questa versione)."
        )

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if (embeddings[i]["subject"] != embeddings[j]["subject"]
                    and embeddings[i].get("dataset") == embeddings[j].get("dataset")):
                all_impostor.append((i, j))   # impostor SOLO dello stesso dataset

    rng.shuffle(all_impostor)

    if max_impostor_pairs is not None:
        impostor = all_impostor[:int(max_impostor_pairs)]
    else:
        impostor = all_impostor

    return genuine, impostor, protocol_used


# ============================================================
# METRICHE VERIFICATION (invariate)
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
    f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) else 0.0

    return {
        "threshold": float(threshold),
        "far": far, "frr": frr, "tpr": tpr, "tnr": tnr,
        "accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
        "precision": precision, "f1": f1,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
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

    g_sorted = np.sort(genuine)
    i_sorted = np.sort(impostor)
    # FAR = frazione impostor >= t ; FRR = frazione genuine < t
    fars = (len(i_sorted) - np.searchsorted(i_sorted, thresholds, side="left")) / len(i_sorted)
    frrs = np.searchsorted(g_sorted, thresholds, side="left") / len(g_sorted)

    idx = int(np.argmin(np.abs(fars - frrs)))
    return {
        "eer": float((fars[idx] + frrs[idx]) / 2.0),
        "eer_threshold": float(thresholds[idx]),
        "eer_far": float(fars[idx]), "eer_frr": float(frrs[idx]),
    }


def tar_at_far(genuine, impostor, target_far):
    impostor = np.asarray(impostor, dtype=np.float64)
    genuine = np.asarray(genuine, dtype=np.float64)
    if len(impostor) == 0:
        return {"target_far": target_far, "threshold": float("nan"),
                "actual_far": float("nan"), "tar": float("nan")}

    g_sorted = np.sort(genuine)
    i_sorted = np.sort(impostor)
    cand = np.concatenate([np.unique(impostor), [impostor.max() + 1e-8]])
    far = (len(i_sorted) - np.searchsorted(i_sorted, cand, side="left")) / len(i_sorted)
    tar = (len(g_sorted) - np.searchsorted(g_sorted, cand, side="left")) / len(g_sorted)

    ok = far <= target_far
    if not ok.any():
        t = cand[-1]
        return {"target_far": target_far, "threshold": float(t),
                "actual_far": float(far[-1]), "tar": float(tar[-1])}
    best = tar[ok].max()
    k = np.where(ok & (tar == best))[0][-1]   # a parità di TAR, soglia più alta (come prima)
    return {"target_far": target_far, "threshold": float(cand[k]),
            "actual_far": float(far[k]), "tar": float(best)}



def evaluate_verification(genuine, impostor, fixed_threshold, name,
                           score_scale_note=None):
    eer = eer_metrics(genuine, impostor)
    fixed = rates_at_threshold(genuine, impostor, fixed_threshold)
    at_eer = rates_at_threshold(genuine, impostor, eer["eer_threshold"])

    result = {
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
    if score_scale_note:
        result["score_scale_note"] = score_scale_note
    return result


# ============================================================
# SCORE EXTRACTION & NORMALIZATION (invariato)
# ============================================================

def compute_score_stats(embeddings, seed=42):
    genuine, impostor, _ = make_verification_pairs(
        embeddings, seed=seed
    )
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


def pair_scores(embeddings, pairs, alpha, norm_stats=None, chunk=500_000):
    P = np.stack([np.asarray(e["palm_embedding"], dtype=np.float32).ravel() for e in embeddings])
    D = np.stack([np.asarray(e["dorsal_embedding"], dtype=np.float32).ravel() for e in embeddings])
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-12

    pairs = np.asarray(pairs, dtype=np.int64)
    sp = np.empty(len(pairs)); sd = np.empty(len(pairs))
    for s in range(0, len(pairs), chunk):
        a, b = pairs[s:s+chunk, 0], pairs[s:s+chunk, 1]
        sp[s:s+chunk] = np.einsum("ij,ij->i", P[a], P[b])
        sd[s:s+chunk] = np.einsum("ij,ij->i", D[a], D[b])

    if norm_stats is not None:
        sp_f = (sp - norm_stats["palm"]["mean"]) / norm_stats["palm"]["std"]
        sd_f = (sd - norm_stats["dorsal"]["mean"]) / norm_stats["dorsal"]["std"]
    else:
        sp_f, sd_f = sp, sd
    return {"palm": sp, "dorsal": sd, "fused": alpha * sp_f + (1.0 - alpha) * sd_f}


# ============================================================
# ALPHA CALIBRATION - K-FOLD (sostituisce lo split singolo 20/80)
# ============================================================

def calibrate_alpha_kfold(embeddings, subjects, k=5, seed=42, use_score_norm=False):
    subjects = sorted(set(subjects))
    if len(subjects) < k * 3:
        raise RuntimeError(
            f"Servono almeno {k * 3} soggetti per {k}-fold "
            f"(disponibili: {len(subjects)}). Riduci k o aggiungi soggetti."
        )

    rng = random.Random(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    folds = [list(f) for f in np.array_split(shuffled, k)]

    grid = np.linspace(0.0, 1.0, 21)
    eer_per_alpha = {float(a): [] for a in grid}
    fold_reports = []

    for fold_idx in range(k):
        cal_subjects = set(folds[fold_idx])
        cal_emb = [x for x in embeddings if x["subject"] in cal_subjects]

        if len(cal_emb) < 4:
            print(f"[CALIBRAZIONE] Fold {fold_idx}: troppo pochi campioni, salto.")
            continue

        genuine, impostor, protocol = make_verification_pairs(
            cal_emb, seed=seed + fold_idx
        )
        norm_stats = compute_score_stats(cal_emb, seed=seed) if use_score_norm else None

        fold_eers = {}
        for alpha in grid:
            scores = pair_scores(cal_emb, genuine + impostor, float(alpha), norm_stats=norm_stats)
            n_g = len(genuine)
            eer = eer_metrics(scores["fused"][:n_g], scores["fused"][n_g:])["eer"]
            eer_per_alpha[float(alpha)].append(eer)
            fold_eers[float(alpha)] = eer

        fold_reports.append({
            "fold": fold_idx,
            "n_subjects": len(cal_subjects),
            "protocol_used": protocol,
            "best_alpha_this_fold": min(fold_eers, key=fold_eers.get),
        })

    mean_eer_by_alpha = {a: float(np.mean(v)) for a, v in eer_per_alpha.items() if v}
    if not mean_eer_by_alpha:
        raise RuntimeError("Calibrazione fallita: nessun fold valido.")

    best_alpha = min(mean_eer_by_alpha, key=mean_eer_by_alpha.get)

    return {
        "alpha": best_alpha,
        "mean_eer_at_best_alpha": mean_eer_by_alpha[best_alpha],
        "mean_eer_by_alpha": mean_eer_by_alpha,  # curva completa, per ispezione plateau
        "k_folds": k,
        "fold_reports": fold_reports,
        "n_subjects_total": len(subjects),
    }


# ============================================================
# ALPHA CALIBRATION PER IL RANKING (Rank-1), separata da quella per EER
# ============================================================

def calibrate_alpha_for_rank(embeddings, subjects, k=5, seed=42, gallery_holdout_real=1,
                             norm_stats=None):
    """
    L'alpha che minimizza l'EER in verifica 1:1 non e' detto sia l'alpha che
    massimizza il Rank-1 in identificazione 1:N: la verifica separa due
    distribuzioni, l'identificazione deve invece ordinare correttamente N
    candidati. Questa funzione fa una grid search subject-disjoint su alpha
    massimizzando il Rank-1 medio (fused) sui k fold.

    COERENZA DI SCALA (fix v4.1): se la valutazione finale usa punteggi
    z-normalizzati (--score_norm zscore), anche la calibrazione DEVE usare gli
    stessi norm_stats. In precedenza qui si usava il coseno grezzo e poi l'alpha
    scelto veniva applicato a punteggi z-normalizzati (scala diversa): l'alpha
    risultava sbilanciato (es. 0.10) e la fusione peggiore del solo palmo.
    """
    subjects = sorted(set(subjects))
    if len(subjects) < k * 3:
        raise RuntimeError(
            f"Servono almeno {k * 3} soggetti per {k}-fold (disponibili: {len(subjects)}). "
            "Riduci --rank_kfold o aggiungi soggetti."
        )

    rng = random.Random(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    folds = [list(f) for f in np.array_split(shuffled, k)]

    grid = np.linspace(0.0, 1.0, 21)
    rank1_per_alpha = {float(a): [] for a in grid}
    fold_reports = []

    for fold_idx in range(k):
        cal_subjects = set(folds[fold_idx])
        cal_emb = [x for x in embeddings if x["subject"] in cal_subjects]

        if len(cal_emb) < 4:
            print(f"[CALIBRAZIONE RANK] Fold {fold_idx}: troppo pochi campioni, salto.")
            continue

        gallery_map, probes = build_gallery_and_probes(cal_emb, holdout_real=gallery_holdout_real)
        if not probes:
            print(f"[CALIBRAZIONE RANK] Fold {fold_idx}: nessun probe disponibile, salto.")
            continue

        fold_rank1 = {}
        for alpha in grid:
            metrics, _ = identification_same_hand(
                gallery_map, probes, float(alpha), "fused_calib", norm_stats=norm_stats
            )
            rank1_per_alpha[float(alpha)].append(metrics["rank1"])
            fold_rank1[float(alpha)] = metrics["rank1"]

        fold_reports.append({
            "fold": fold_idx,
            "n_subjects": len(cal_subjects),
            "n_probes": len(probes),
            "best_alpha_this_fold": max(fold_rank1, key=fold_rank1.get),
        })

    mean_rank1_by_alpha = {a: float(np.mean(v)) for a, v in rank1_per_alpha.items() if v}
    if not mean_rank1_by_alpha:
        raise RuntimeError("Calibrazione alpha per rank fallita: nessun fold valido.")

    best_alpha = max(mean_rank1_by_alpha, key=mean_rank1_by_alpha.get)

    return {
        "alpha": best_alpha,
        "mean_rank1_at_best_alpha": mean_rank1_by_alpha[best_alpha],
        "mean_rank1_by_alpha": mean_rank1_by_alpha,
        "score_normalization_used": "zscore" if norm_stats is not None else "none",
        "k_folds": k,
        "fold_reports": fold_reports,
        "n_subjects_total": len(subjects),
    }


# ============================================================
# IDENTIFICATION 1:N (invariato nella logica, solo pulizia)
# ============================================================

def identification_same_hand(gallery_map, probes, alpha, name, norm_stats=None,
                              debug_tie_threshold=1e-6, verbose_debug=False):
    """
    NOTA: la galleria confronta il probe SOLO con le identita' della stessa
    mano (side) del probe. Confrontare anche con le gallery dell'altra mano
    (L vs R) e' inutile ai fini del ranking (non puo' mai essere la risposta
    corretta nel protocollo "same hand") e nella versione precedente veniva
    fatto comunque, sprecando calcolo e rendendo piu' difficile diagnosticare
    i pareggi di punteggio.
    """
    details = []
    rank1 = rank5 = rank10 = 0
    reciprocal_sum = 0.0
    n_tied_top1 = 0

    for true_label, probe in probes:
        probe_side = true_label.rsplit("_", 1)[-1]
        candidates = {
            gal_label: gal for gal_label, gal in gallery_map.items()
            if gal_label.rsplit("_", 1)[-1] == probe_side
            and _label_dataset(gal_label) == _label_dataset(true_label)
        }

        scores = []
        for gal_label, gal in candidates.items():
            sp = cosine(probe["palm_embedding"], gal["palm_embedding"])
            sd = cosine(probe["dorsal_embedding"], gal["dorsal_embedding"])
            if norm_stats is not None:
                sp_fusion = _normalize(sp, norm_stats["palm"])
                sd_fusion = _normalize(sd, norm_stats["dorsal"])
            else:
                sp_fusion, sd_fusion = sp, sd
            sf = alpha * sp_fusion + (1.0 - alpha) * sd_fusion
            scores.append({"label": gal_label, "palm_similarity": sp,
                            "dorsal_similarity": sd, "fused_similarity": sf})

        scores.sort(key=lambda x: x["fused_similarity"], reverse=True)
        ranked_labels = [x["label"] for x in scores]
        rank = ranked_labels.index(true_label) + 1

        # Margine tra il candidato migliore e il secondo: se e' piccolo (anche
        # quando rank==1) la separazione e' fragile; se e' grande solo quando
        # rank>1 il problema e' probabilmente localizzato su pochi soggetti
        # "difficili" piuttosto che strutturale su tutto l'embedding space.
        margin_top1_top2 = (
            scores[0]["fused_similarity"] - scores[1]["fused_similarity"]
            if len(scores) > 1 else float("nan")
        )
        if len(scores) > 1 and abs(margin_top1_top2) < debug_tie_threshold:
            n_tied_top1 += 1

        rank1 += int(rank <= 1)
        rank5 += int(rank <= 5)
        rank10 += int(rank <= 10)
        reciprocal_sum += 1.0 / rank

        details.append({
            "system": name, "probe_subject": probe["subject"], "probe_side": probe["side"],
            "gallery_side": probe["side"], "rank": rank,
            "predicted_subject": ranked_labels[0],
            "top1_correct": bool(rank == 1), "top5_correct": bool(rank <= 5),
            "top10_correct": bool(rank <= 10),
            "top1_palm": max(scores, key=lambda x: x["palm_similarity"])["label"],
            "top1_dorsal": max(scores, key=lambda x: x["dorsal_similarity"])["label"],
            "margin_top1_top2": margin_top1_top2,
            "probe_is_augmented": bool(probe.get("is_augmented", False)),
        })

    n = len(probes)
    n_real = sum(1 for _, p in probes if not p.get("is_augmented", False))
    n_aug = n - n_real
    rank1_real = sum(1 for d in details if d["top1_correct"] and not d["probe_is_augmented"])
    rank1_aug = sum(1 for d in details if d["top1_correct"] and d["probe_is_augmented"])
    margins = [d["margin_top1_top2"] for d in details if not np.isnan(d["margin_top1_top2"])]

    if n_tied_top1 > 0:
        print(
            f"  [{name}] ATTENZIONE: {n_tied_top1}/{n} query hanno un pareggio "
            f"(quasi-)esatto tra top1 e top2 (|margine| < {debug_tie_threshold}). "
            "Questo di solito indica embedding non discriminativi tra soggetti "
            "diversi (collasso), oppure un bug a monte nell'estrazione/caching "
            "degli embedding: vale la pena controllare quei casi prima di "
            "toccare alpha o la galleria."
        )

    return {
        "system": name, "gallery_side": "Same Hand (Enrollment)",
        "probe_side": "Same Hand (Held-out)",
        "n_gallery_subjects": len(gallery_map), "n_probes": n,
        "n_probes_real": n_real, "n_probes_augmented": n_aug,
        "rank1": rank1 / n if n else 0.0, "rank1_percent": 100.0 * rank1 / n if n else 0.0,
        "rank1_percent_real_probes": 100.0 * rank1_real / n_real if n_real else None,
        "rank1_percent_augmented_probes": 100.0 * rank1_aug / n_aug if n_aug else None,
        "rank5": rank5 / n if n else 0.0, "rank5_percent": 100.0 * rank5 / n if n else 0.0,
        "rank10": rank10 / n if n else 0.0, "rank10_percent": 100.0 * rank10 / n if n else 0.0,
        "mrr": reciprocal_sum / n if n else 0.0,
        "mean_margin_top1_top2": float(np.mean(margins)) if margins else None,
        "mean_margin_when_wrong": float(np.mean(
            [d["margin_top1_top2"] for d in details if not d["top1_correct"] and not np.isnan(d["margin_top1_top2"])]
        )) if any(not d["top1_correct"] for d in details) else None,
        "n_tied_top1": n_tied_top1,
        "n_tied_top1_percent": 100.0 * n_tied_top1 / n if n else 0.0,
    }, details


def _is_aug_sample(x):
    # sample_id include sempre il tag "aug<N>" per i campioni generati
    # automaticamente (vedi discover_raw_samples/_build_augmented_filename).
    return "aug" in x["sample_id"]


def _mean_embedding(samples, key):
    """
    Media degli embedding sulla SFERA UNITARIA (L2-normalizzati prima e dopo
    la media), non nello spazio grezzo.

    Perche': se gli embedding hanno norme diverse tra loro, una media
    aritmetica semplice e' dominata dai vettori a norma piu' alta e produce
    un prototipo di galleria "sbilanciato" verso quei campioni, il che
    appiattisce le differenze tra soggetti proprio nel confronto coseno
    usato per il ranking (sintomo tipico: margine top1-top2 ~ 0 in tutte le
    query, come osservato). Normalizzare prima e dopo la media rende il
    prototipo un vero "centroide direzionale", piu' stabile e discriminativo.
    """
    normed = [s[key] / s[key].norm().clamp_min(1e-12) for s in samples]
    stacked = torch.stack(normed, dim=0)
    mean = stacked.mean(dim=0)
    return mean / mean.norm().clamp_min(1e-12)


def build_gallery_and_probes(embeddings, holdout_real=1):
    """
    Costruisce gallery ed elenco probe per l'identificazione 1:N.

    Novita' rispetto alla v3 (gallery = solo il primo scatto):
    - la gallery di ogni identita' (subject, side) e' la MEDIA degli embedding
      di tutti gli scatti REALI (non augmentati) tranne `holdout_real` di essi,
      che vengono invece tenuti come probe. Una gallery mediata su piu' scatti
      reali e' un enrollment meno rumoroso di un singolo scatto.
    - se una mano ha un solo scatto reale, quello resta l'unico enrollment
      (impossibile fare holdout senza svuotare la gallery) e i probe per
      quella mano sono solo gli scatti augmentati.
    - ogni probe porta con se' il flag is_augmented, per poter separare le
      metriche "vere" (probe reali) da quelle sugli scatti sintetici.
    """
    by_subj_side = {}
    for x in embeddings:
        by_subj_side.setdefault((x["subject"], x["side"]), []).append(x)

    gallery_map = {}
    probes = []

    for (subj, side), samples in sorted(by_subj_side.items()):
        identity_label = f"{subj}_{side}"
        real_samples = [s for s in samples if not _is_aug_sample(s)]
        aug_samples = [s for s in samples if _is_aug_sample(s)]

        if len(real_samples) > holdout_real:
            # abbastanza scatti reali per tenerne alcuni fuori dalla gallery
            enrollment = real_samples[:-holdout_real] if holdout_real > 0 else real_samples
            held_out_real = real_samples[len(enrollment):]
        else:
            enrollment = real_samples
            held_out_real = []

        if not enrollment:
            # nessuno scatto reale disponibile: fallback al primo scatto
            # disponibile (reale o augmentato) come singolo enrollment
            enrollment = samples[:1]

        gallery_map[identity_label] = {
            "palm_embedding": _mean_embedding(enrollment, "palm_embedding"),
            "dorsal_embedding": _mean_embedding(enrollment, "dorsal_embedding"),
            "n_enrollment_samples": len(enrollment),
        }

        for probe_sample in held_out_real:
            p = dict(probe_sample)
            p["is_augmented"] = False
            probes.append((identity_label, p))
        for probe_sample in aug_samples:
            p = dict(probe_sample)
            p["is_augmented"] = True
            probes.append((identity_label, p))

    return gallery_map, probes


def identification_all_systems(embeddings, alpha, norm_stats=None, gallery_holdout_real=1):
    by_subj_side = {}
    for x in embeddings:
        by_subj_side.setdefault((x["subject"], x["side"]), []).append(x)

    has_multi_samples = any(len(samples) >= 2 for samples in by_subj_side.values())

    if not has_multi_samples:
        raise RuntimeError(
            "Identificazione 1:N: nessuna mano con almeno 2 scatti reali disponibile."
        )

    print(
        "\n[IDENTIFICAZIONE 1:N] Protocollo Stessa Mano "
        "(Gallery: media scatti reali enrollment, Probe: scatti reali held-out + augmentati)"
    )
    all_metrics = []
    all_details = []

    gallery_map, probes = build_gallery_and_probes(embeddings, holdout_real=gallery_holdout_real)

    for system, a in [("palm", 1.0), ("dorsal", 0.0), ("fused", alpha)]:
        stats_for_call = norm_stats if system == "fused" else None
        metrics, details = identification_same_hand(
            gallery_map, probes, a, system, norm_stats=stats_for_call
        )
        all_metrics.append(metrics)
        all_details.extend(details)

    protocol_used = "same_hand_multi_sample_meanenroll"
    return all_metrics, all_details, protocol_used


# ============================================================
# IDENTIFICAZIONE OPEN-SET (v4)
# ============================================================
#
# Perche' serve: nell'identificazione closed-set (sopra) il probe appartiene
# SEMPRE a una identita' in gallery, quindi il sistema non deve mai rispondere
# "sconosciuto". In un uso reale arrivano anche persone NON iscritte: il sistema
# deve (a) rifiutarle e (b) identificare correttamente quelle iscritte.
#
# Protocollo (subject-disjoint, ripetuto su piu' split casuali):
#   - i soggetti del NUOVO dataset vengono divisi in
#         KNOWN        -> iscritti in gallery (media degli scatti reali di enrollment)
#         UNKNOWN_CAL  -> mai iscritti, usati SOLO per scegliere la soglia
#         UNKNOWN_TEST -> mai iscritti, usati SOLO per misurare FPIR
#   - probe genuini  = scatti held-out (reali e/o augmentati) dei soggetti KNOWN
#   - probe impostori = scatti dei soggetti UNKNOWN
#   - decisione: si prende il miglior punteggio in gallery (stessa mano del probe);
#     se >= soglia t -> "identificato come quel candidato", altrimenti "sconosciuto".
#
# Perche' la soglia si calibra sul NUOVO dataset: i checkpoint sono addestrati su un
# dataset diverso, quindi la distribuzione dei punteggi (coseno) cambia (domain
# shift) e le soglie ottenute sul dataset di training NON sono trasferibili. Qui la
# soglia viene fissata a un FPIR-obiettivo su soggetti UNKNOWN_CAL e poi valutata su
# soggetti UNKNOWN_TEST DISGIUNTI: il FPIR ottenuto su UNKNOWN_TEST dice quanto la
# calibrazione regge davvero. Come riferimento ottimistico si riporta anche la
# soglia "oracle" (scelta direttamente su UNKNOWN_TEST).
#
# Metriche (stile ISO/IEC 19795-1):
#   FPIR  = frazione di probe unknown accettati (top-1 >= t)
#   FNIR  = 1 - DIR@1(t)
#   DIR@k = frazione di probe genuini con identita' corretta entro il rank k e con
#           punteggio >= t   (DIR@1 = "correttamente identificato E accettato")
#   FRR_known     = genuini rifiutati come sconosciuti (top-1 < t)
#   misidentified = genuini accettati ma assegnati all'identita' sbagliata
#   Open-set EER  = punto in cui FPIR == FNIR
#   detection AUC = AUROC "known vs unknown" sul miglior punteggio in gallery
#
# ATTENZIONE: i probe genuini augmentati derivano dalla STESSA immagine usata per
# l'enrollment, quindi sono facili e ottimistici. Le metriche vengono riportate
# sia su "all" (reali+augmentati) sia su "real_only" (solo scatti reali held-out,
# che richiedono >= 2 scatti reali per mano): il numero da citare e' real_only.

def _to_unit_np(t):
    v = t.detach().cpu().float().numpy().astype(np.float64).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def score_matrix_vs_gallery(gallery_map, probes, alpha, norm_stats=None):
    """
    Matrice [n_probe, n_gallery] di punteggi fusi (alpha*palmo + (1-alpha)*dorso,
    z-normalizzati se norm_stats). I candidati di gallery con side diverso dal
    probe ricevono -inf (stessa regola dell'identificazione closed-set).
    """
    gal_labels = sorted(gallery_map.keys())
    gal_sides = np.array([l.rsplit("_", 1)[-1] for l in gal_labels])
    if not probes:
        return np.zeros((0, len(gal_labels))), gal_labels

    gp = np.stack([_to_unit_np(gallery_map[l]["palm_embedding"]) for l in gal_labels])
    gd = np.stack([_to_unit_np(gallery_map[l]["dorsal_embedding"]) for l in gal_labels])
    pp = np.stack([_to_unit_np(p["palm_embedding"]) for _, p in probes])
    pdor = np.stack([_to_unit_np(p["dorsal_embedding"]) for _, p in probes])

    sp = pp @ gp.T
    sd = pdor @ gd.T
    if norm_stats is not None:
        sp = _normalize(sp, norm_stats["palm"])
        sd = _normalize(sd, norm_stats["dorsal"])

    fused = alpha * sp + (1.0 - alpha) * sd
    probe_sides = np.array([p["side"] for _, p in probes])
    fused = np.where(probe_sides[:, None] == gal_sides[None, :], fused, -np.inf)
    gal_ds = np.array([_label_dataset(l) for l in gal_labels])
    probe_ds = np.array([_label_dataset(lbl) for lbl, _ in probes])
    fused = np.where(probe_ds[:, None] == gal_ds[None, :], fused, -np.inf)
    return fused, gal_labels


def _genuine_stats(S, gal_labels, probes):
    n = len(probes)
    if n == 0:
        z = np.zeros(0)
        return {"top1": z, "true_score": z, "rank": z.astype(int),
                "is_aug": z.astype(bool), "top1_idx": z.astype(int)}
    index = {l: i for i, l in enumerate(gal_labels)}
    true_idx = np.array([index[lbl] for lbl, _ in probes])
    true_score = S[np.arange(n), true_idx]
    rank = 1 + (S > true_score[:, None]).sum(axis=1)
    return {
        "top1": S.max(axis=1),
        "true_score": true_score,
        "rank": rank.astype(int),
        "is_aug": np.array([bool(p.get("is_augmented", False)) for _, p in probes]),
        "top1_idx": S.argmax(axis=1),
    }


def _subset(stats, mask):
    return {k: v[mask] for k, v in stats.items()}


def _fpir(unk_top1, t):
    if len(unk_top1) == 0 or np.isnan(t):
        return float("nan")
    return float(np.mean(unk_top1 >= t))


def threshold_at_fpir(unk_top1, target):
    """Soglia piu' bassa (=> DIR massimo) tale che FPIR(t) <= target sugli unknown dati."""
    n = len(unk_top1)
    if n == 0:
        return float("nan")
    u = np.sort(unk_top1[np.isfinite(unk_top1)])[::-1]
    if len(u) == 0:
        return float("nan")
    k = int(np.floor(target * n))
    if k >= len(u):
        return float(u[-1]) - 1e-7
    return float(np.nextafter(u[k], np.inf))


def open_set_report_at_threshold(gen, unk_top1, t):
    n_g = len(gen["rank"])
    if n_g == 0 or np.isnan(t):
        return {"threshold": float(t), "fpir": _fpir(unk_top1, t),
                "dir_rank1": float("nan"), "dir_rank5": float("nan"),
                "fnir": float("nan"), "false_reject_known": float("nan"),
                "misidentified": float("nan")}
    accepted = gen["top1"] >= t
    correct1 = accepted & (gen["rank"] == 1)
    dir5 = (gen["rank"] <= 5) & (gen["true_score"] >= t)
    return {
        "threshold": float(t),
        "fpir": _fpir(unk_top1, t),
        "dir_rank1": float(np.mean(correct1)),
        "dir_rank5": float(np.mean(dir5)),
        "fnir": float(1.0 - np.mean(correct1)),
        "false_reject_known": float(np.mean(~accepted)),
        "misidentified": float(np.mean(accepted & (gen["rank"] > 1))),
    }


def open_set_eer(gen, unk_top1):
    n_g, n_u = len(gen["rank"]), len(unk_top1)
    nan = {"open_set_eer": float("nan"), "open_set_eer_threshold": float("nan")}
    if n_g == 0 or n_u == 0:
        return nan
    vals = np.concatenate([gen["top1"], unk_top1])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return nan

    s = np.unique(vals)
    if len(s) == 1:
        thr = s
    else:
        thr = np.concatenate([[s[0] - 1e-7], (s[:-1] + s[1:]) / 2.0, [s[-1] + 1e-7]])

    sorted_unk = np.sort(unk_top1)
    fpir = 1.0 - np.searchsorted(sorted_unk, thr, side="left") / n_u
    correct_scores = np.sort(gen["true_score"][gen["rank"] == 1])
    dir1 = (len(correct_scores) - np.searchsorted(correct_scores, thr, side="left")) / n_g
    fnir = 1.0 - dir1

    idx = int(np.argmin(np.abs(fpir - fnir)))
    return {"open_set_eer": float((fpir[idx] + fnir[idx]) / 2.0),
            "open_set_eer_threshold": float(thr[idx])}


def open_set_curve(gen, unk_top1, max_points=300):
    if len(gen["rank"]) == 0 or len(unk_top1) == 0:
        return []
    vals = np.concatenate([gen["top1"], unk_top1])
    vals = np.unique(vals[np.isfinite(vals)])
    if len(vals) == 0:
        return []
    if len(vals) > max_points:
        vals = vals[np.linspace(0, len(vals) - 1, max_points).astype(int)]
    rows = []
    for t in vals:
        rep = open_set_report_at_threshold(gen, unk_top1, float(t))
        rows.append({"threshold": float(t), "fpir": rep["fpir"], "fnir": rep["fnir"],
                     "dir_rank1": rep["dir_rank1"], "dir_rank5": rep["dir_rank5"]})
    return rows


def _pct_label(target):
    return f"{target * 100:g}pct"


def _split_subjects_open_set(subjects, seed, known_fraction, cal_fraction):
    """Split known / unknown_cal / unknown_test subject-disjoint, STRATIFICATO per dataset."""
    rng = random.Random(seed)
    groups = {}
    for sub in sorted(subjects):
        groups.setdefault(_label_dataset(sub), []).append(sub)
    known, unk_cal, unk_test = set(), set(), set()
    for ds in sorted(groups):
        subs = groups[ds]
        rng.shuffle(subs)
        if len(subs) < 4:
            raise ValueError(f"Dataset '{ds}': servono >= 4 soggetti per lo split open-set.")
        n_known = min(max(int(round(known_fraction * len(subs))), 2), len(subs) - 2)
        unknown = subs[n_known:]
        n_cal = min(max(int(round(cal_fraction * len(unknown))), 1), len(unknown) - 1)
        known |= set(subs[:n_known])
        unk_cal |= set(unknown[:n_cal])
        unk_test |= set(unknown[n_cal:])
    return known, unk_cal, unk_test


def open_set_single_split(embeddings, subjects, alpha, norm_stats, seed,
                          known_fraction, cal_fraction, target_fpirs,
                          gallery_holdout_real, unknown_samples, collect=False):
    known, unk_cal, unk_test = _split_subjects_open_set(
        subjects, seed, known_fraction, cal_fraction
    )

    known_emb = [x for x in embeddings if x["subject"] in known]
    gallery_map, gen_probes = build_gallery_and_probes(
        known_emb, holdout_real=gallery_holdout_real
    )

    def unknown_probes(sub_set):
        out = []
        for x in embeddings:
            if x["subject"] not in sub_set:
                continue
            aug = _is_aug_sample(x)
            if unknown_samples == "real" and aug:
                continue
            p = dict(x)
            p["is_augmented"] = aug
            out.append((f"{x['subject']}_{x['side']}", p))
        return out

    cal_probes = unknown_probes(unk_cal)
    test_probes = unknown_probes(unk_test)

    result = {"sizes": {
        "n_known_subjects": len(known), "n_unknown_cal_subjects": len(unk_cal),
        "n_unknown_test_subjects": len(unk_test), "n_gallery_identities": len(gallery_map),
        "n_genuine_probes": len(gen_probes), "n_unknown_cal_probes": len(cal_probes),
        "n_unknown_test_probes": len(test_probes),
    }}
    curve_rows, detail_rows = [], []

    for system, a in [("palm", 1.0), ("dorsal", 0.0), ("fused", alpha)]:
        ns = norm_stats if system == "fused" else None

        S_gen, gal_labels = score_matrix_vs_gallery(gallery_map, gen_probes, a, ns)
        S_cal, _ = score_matrix_vs_gallery(gallery_map, cal_probes, a, ns)
        S_test, _ = score_matrix_vs_gallery(gallery_map, test_probes, a, ns)

        gen = _genuine_stats(S_gen, gal_labels, gen_probes)
        cal_top1 = S_cal.max(axis=1) if len(cal_probes) else np.zeros(0)
        test_top1 = S_test.max(axis=1) if len(test_probes) else np.zeros(0)

        result[system] = {}
        for ps_name, g in (("all", gen), ("real_only", _subset(gen, ~gen["is_aug"]))):
            entry = {"n_genuine_probes": len(g["rank"])}
            if len(g["rank"]) == 0:
                entry.update({"rank1_known": float("nan"),
                              "detection_auc": float("nan"),
                              "open_set_eer": float("nan"),
                              "open_set_eer_threshold": float("nan")})
            else:
                entry["rank1_known"] = float(np.mean(g["rank"] == 1))
                entry["detection_auc"] = roc_auc(g["top1"], test_top1)
                entry.update(open_set_eer(g, test_top1))

            entry["oracle"], entry["calibrated"] = {}, {}
            for target in target_fpirs:
                lab = _pct_label(target)
                t_or = threshold_at_fpir(test_top1, target)
                entry["oracle"][f"fpir_{lab}"] = open_set_report_at_threshold(g, test_top1, t_or)

                t_cal = threshold_at_fpir(cal_top1, target)
                rep = open_set_report_at_threshold(g, test_top1, t_cal)
                rep["fpir_on_calibration_set"] = _fpir(cal_top1, t_cal)
                entry["calibrated"][f"fpir_{lab}"] = rep

            result[system][ps_name] = entry

            if collect:
                all_unk = np.concatenate([cal_top1, test_top1])
                for row in open_set_curve(g, all_unk):
                    curve_rows.append({"system": system, "probe_set": ps_name, **row})

        if collect:
            def add_rows(role, S, probes, stats=None):
                if not len(probes):
                    return
                top_idx = S.argmax(axis=1)
                for i, (lbl, p) in enumerate(probes):
                    detail_rows.append({
                        "system": system, "role": role, "probe_identity": lbl,
                        "probe_sample_id": p.get("sample_id"),
                        "probe_is_augmented": bool(p.get("is_augmented", False)),
                        "top1_identity": gal_labels[int(top_idx[i])],
                        "top1_score": float(S[i, top_idx[i]]),
                        "true_score": float(stats["true_score"][i]) if stats is not None else "",
                        "rank_true_identity": int(stats["rank"][i]) if stats is not None else "",
                    })
            add_rows("genuine_known", S_gen, gen_probes, gen)
            add_rows("unknown_cal", S_cal, cal_probes)
            add_rows("unknown_test", S_test, test_probes)

    return result, curve_rows, detail_rows


def _aggregate_nested(list_of_dicts):
    keys = sorted(set().union(*[d.keys() for d in list_of_dicts]))
    out = {}
    for k in keys:
        vals = [d[k] for d in list_of_dicts if k in d]
        if all(isinstance(v, dict) for v in vals):
            out[k] = _aggregate_nested(vals)
        else:
            nums = []
            for v in vals:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fv):
                    nums.append(fv)
            out[k] = {
                "mean": float(np.mean(nums)) if nums else None,
                "std": float(np.std(nums)) if nums else None,
                "n": len(nums),
            }
    return out


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _fmt_ms(leaf, scale=100.0, digits=2):
    if not leaf or leaf.get("mean") is None:
        return "n/d"
    return f"{leaf['mean'] * scale:.{digits}f}±{leaf['std'] * scale:.{digits}f}"


def format_open_set_summary(res):
    p = res["protocol"]
    agg = res["aggregate"]
    lines = [
        f"Split ripetuti: {p['n_repeats']} | soggetti totali: {p['n_subjects']} | "
        f"known={p['n_known_subjects']}, unknown_cal={p['n_unknown_cal_subjects']}, "
        f"unknown_test={p['n_unknown_test_subjects']} (per split)",
        f"Probe unknown: scatti '{p['unknown_samples']}' | alpha fusione: {p['alpha']:.2f} | "
        f"score_norm: {p['score_normalization']}",
        "Soglia = FPIR-obiettivo calibrato su unknown_cal, valutato su unknown_test "
        "(soggetti disgiunti). [oracle] = soglia scelta su unknown_test (ottimistica).",
        "Valori: media±std sugli split, in %.",
    ]
    labels = {"all": "tutti i probe genuini (nessuna augmentation)", "real_only": "probe genuini SOLO reali"}
    for system in ("palm", "dorsal", "fused"):
        if system not in agg:
            continue
        for ps in ("all", "real_only"):
            e = agg[system].get(ps)
            if e is None:
                continue
            n_g = e["n_genuine_probes"]["mean"]
            lines.append("")
            lines.append(f"[{system.upper()}] {labels[ps]} (~{n_g:.0f} probe genuini/split)")
            if n_g is None or n_g == 0:
                lines.append("   n/d: nessun probe genuino reale held-out "
                             "(servono >= 2 scatti REALI per mano)")
                continue
            lines.append(
                f"   Rank-1 (solo known)={_fmt_ms(e['rank1_known'])} | "
                f"AUC known-vs-unknown={_fmt_ms(e['detection_auc'], 1.0, 4)} | "
                f"Open-set EER={_fmt_ms(e['open_set_eer'])}"
            )
            for lab in sorted(e["calibrated"].keys(), key=lambda s: float(s.split("_")[1][:-3])):
                c = e["calibrated"][lab]
                o = e["oracle"][lab]
                lines.append(
                    f"   FPIR-obiettivo {lab.split('_')[1][:-3]}%: "
                    f"FPIR_test={_fmt_ms(c['fpir'])} | DIR@1={_fmt_ms(c['dir_rank1'])} | "
                    f"FRR_known={_fmt_ms(c['false_reject_known'])} | "
                    f"misid={_fmt_ms(c['misidentified'])} | "
                    f"soglia={_fmt_ms(c['threshold'], 1.0, 3)} | "
                    f"[oracle DIR@1={_fmt_ms(o['dir_rank1'])}]"
                )
    return lines


def evaluate_open_set(embeddings, alpha, norm_stats, args, results_dir):
    subjects = sorted(set(x["subject"] for x in embeddings))
    if len(subjects) < 4:
        print("[OPEN-SET] Servono almeno 4 soggetti (>=2 known e >=2 unknown): salto.")
        return None
    if not 0.0 < args.openset_known_fraction < 1.0:
        raise ValueError("--openset_known_fraction deve essere in (0, 1).")
    if not 0.0 < args.openset_cal_fraction < 1.0:
        raise ValueError("--openset_cal_fraction deve essere in (0, 1).")

    targets = sorted(float(t) for t in str(args.openset_target_fpirs).split(",") if t.strip())
    if not targets or any(not 0.0 < t < 1.0 for t in targets):
        raise ValueError("--openset_target_fpirs: valori in (0, 1) separati da virgola.")

    per_split, first_sizes = [], None
    for r in range(args.openset_repeats):
        res, curve_rows, detail_rows = open_set_single_split(
            embeddings, subjects, alpha, norm_stats,
            seed=args.seed + 1000 + r,
            known_fraction=args.openset_known_fraction,
            cal_fraction=args.openset_cal_fraction,
            target_fpirs=targets,
            gallery_holdout_real=args.gallery_holdout_real,
            unknown_samples=args.openset_unknown_samples,
            collect=(r == 0),
        )
        per_split.append(res)
        if r == 0:
            first_sizes = res["sizes"]
            with open(results_dir / "open_set_curve_split0.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["system", "probe_set", "threshold", "fpir",
                                                  "fnir", "dir_rank1", "dir_rank5"])
                w.writeheader()
                w.writerows(curve_rows)
            with open(results_dir / "open_set_details_split0.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "system", "role", "probe_identity", "probe_sample_id", "probe_is_augmented",
                    "top1_identity", "top1_score", "true_score", "rank_true_identity"])
                w.writeheader()
                w.writerows(detail_rows)

    aggregate = _aggregate_nested(per_split)

    out = {
        "protocol": {
            "n_repeats": args.openset_repeats,
            "n_subjects": len(subjects),
            "n_known_subjects": first_sizes["n_known_subjects"],
            "n_unknown_cal_subjects": first_sizes["n_unknown_cal_subjects"],
            "n_unknown_test_subjects": first_sizes["n_unknown_test_subjects"],
            "known_fraction": args.openset_known_fraction,
            "cal_fraction": args.openset_cal_fraction,
            "target_fpirs": targets,
            "unknown_samples": args.openset_unknown_samples,
            "gallery_holdout_real": args.gallery_holdout_real,
            "alpha": alpha,
            "score_normalization": args.score_norm,
            "threshold_policy": "calibrata su unknown_cal (FPIR-obiettivo), valutata su unknown_test disgiunti",
        },
        "aggregate": aggregate,
        "per_split": per_split,
    }
    save_json(results_dir / "open_set_metrics.json", _json_safe(out))

    for line in format_open_set_summary(out):
        print(line)
    return out


# ============================================================
# OUTPUT (invariato tranne aggiunta protocollo)
# ============================================================

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_verification_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "system", "n_genuine", "n_impostor", "eer", "eer_percent",
            "eer_threshold", "roc_auc", "fixed_threshold", "fixed_accuracy",
            "fixed_balanced_accuracy", "fixed_far", "fixed_frr", "fixed_precision",
            "fixed_f1", "eer_accuracy", "eer_balanced_accuracy", "eer_far",
            "eer_frr", "eer_precision", "eer_f1", "tar_at_far_1pct",
            "tar_at_far_5pct", "tar_at_far_10pct",
        ])
        writer.writeheader()
        for r in rows:
            fixed = r["fixed_threshold_metrics"]
            eerm = r["eer_threshold_metrics"]
            writer.writerow({
                "system": r["system"], "n_genuine": r["n_genuine"], "n_impostor": r["n_impostor"],
                "eer": r["eer"], "eer_percent": r["eer_percent"], "eer_threshold": r["eer_threshold"],
                "roc_auc": r["roc_auc"], "fixed_threshold": fixed["threshold"],
                "fixed_accuracy": fixed["accuracy"], "fixed_balanced_accuracy": fixed["balanced_accuracy"],
                "fixed_far": fixed["far"], "fixed_frr": fixed["frr"], "fixed_precision": fixed["precision"],
                "fixed_f1": fixed["f1"], "eer_accuracy": eerm["accuracy"],
                "eer_balanced_accuracy": eerm["balanced_accuracy"], "eer_far": eerm["far"],
                "eer_frr": eerm["frr"], "eer_precision": eerm["precision"], "eer_f1": eerm["f1"],
                "tar_at_far_1pct": r["tar_at_far_1pct"]["tar"],
                "tar_at_far_5pct": r["tar_at_far_5pct"]["tar"],
                "tar_at_far_10pct": r["tar_at_far_10pct"]["tar"],
            })


def save_identification_csv(path, rows):
    # extrasaction="ignore" e fieldnames allineati alle chiavi realmente
    # restituite da identification_same_hand: la versione precedente andava
    # in ValueError perche' i dizionari contenevano piu' chiavi (es.
    # n_probes_real, mean_margin_top1_top2, n_tied_top1, ...) di quelle
    # dichiarate qui.
    fields = ["system", "gallery_side", "probe_side", "n_gallery_subjects", "n_probes",
              "n_probes_real", "n_probes_augmented",
              "rank1", "rank1_percent", "rank1_percent_real_probes", "rank1_percent_augmented_probes",
              "rank5", "rank5_percent", "rank10", "rank10_percent", "mrr",
              "mean_margin_top1_top2", "mean_margin_when_wrong",
              "n_tied_top1", "n_tied_top1_percent"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_details_csv(path, rows):
    fields = ["system", "probe_subject", "probe_side", "gallery_side", "rank",
              "predicted_subject", "top1_correct", "top5_correct", "top10_correct",
              "top1_palm", "top1_dorsal", "margin_top1_top2", "probe_is_augmented"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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
                sp_fusion, sd_fusion = sp, sd
            sf = alpha * sp_fusion + (1.0 - alpha) * sd_fusion
            rows.append({
                "label": label, "subject1": embeddings[i]["subject"], "side1": embeddings[i]["side"],
                "subject2": embeddings[j]["subject"], "side2": embeddings[j]["side"],
                "palm_similarity": sp, "dorsal_similarity": sd, "fused_similarity": sf,
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = ["label", "subject1", "side1", "subject2", "side2",
                  "palm_similarity", "dorsal_similarity", "fused_similarity"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN EVALUATION
# ============================================================

def parse_tag_specs(specs, what):
    out = []
    for sp in specs or []:
        if "=" not in sp:
            raise ValueError(f"{what}: atteso TAG=PATH, ricevuto '{sp}'")
        tag, path = sp.split("=", 1)
        tag, path = tag.strip(), path.strip()
        if not re.fullmatch(r"[A-Za-z0-9\-]+", tag):
            raise ValueError(f"{what}: il TAG '{tag}' puo' contenere solo lettere, cifre e '-'.")
        out.append((tag, path))
    return out


def _tag_pairs(pairs, tag):
    out = []
    for p in pairs:
        q = dict(p)
        q["raw_subject"] = p["subject"]
        q["subject"] = f"{tag}:{p['subject']}"
        q["dataset"] = tag
        q["is_augmented"] = False
        out.append(q)
    return out


def load_preprocessed_dataset(tag, directory, dorsal_directory=None):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"[{tag}] cartella non trovata: {directory}")
    pairs, _records, missing, failures, unparsed = discover_preprocessed_samples(directory, dorsal_directory)
    n_aug = sum(1 for p in pairs if "aug" in p["sample_id"])
    pairs = [p for p in pairs if "aug" not in p["sample_id"]]      # niente augmentation
    info = {"tag": tag, "kind": "preprocessed", "path": str(directory),
            "dorsal_path": str(dorsal_directory) if dorsal_directory is not None else str(directory),
            "n_pairs": len(pairs),
            "n_ignored_augmented": n_aug, "missing_pairs": missing,
            "preprocess_failures": failures, "n_unparsed_files": len(unparsed)}
    return _tag_pairs(pairs, tag), info


def drop_single_shot_hands(pairs, min_shots=2):
    count = {}
    for p in pairs:
        k = (p["subject"], p["side"])
        count[k] = count.get(k, 0) + 1
    keep = [p for p in pairs if count[(p["subject"], p["side"])] >= min_shots]
    dropped = sorted(f"{k[0]}_{k[1]}" for k, c in count.items() if c < min_shots)
    return keep, dropped


def evaluate_dataset(args):
    out_root = Path(args.out_dir)
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.palm_pre or args.dorsal_pre:
        if args.dataset_pre:
            raise ValueError("Usa --dataset_pre oppure la coppia --palm_pre/--dorsal_pre, non entrambi.")
        if not args.palm_pre or not args.dorsal_pre:
            raise ValueError("Con cartelle separate servono sia --palm_pre TAG=DIR sia --dorsal_pre TAG=DIR.")
        palm_specs = parse_tag_specs(args.palm_pre, "--palm_pre")
        dorsal_specs = parse_tag_specs(args.dorsal_pre, "--dorsal_pre")
        if dict(palm_specs).keys() != dict(dorsal_specs).keys():
            raise ValueError("I TAG di --palm_pre e --dorsal_pre devono coincidere.")
        dorsal_by_tag = dict(dorsal_specs)
        specs = [(tag, palm_path, dorsal_by_tag[tag]) for tag, palm_path in palm_specs]
    else:
        specs = [(tag, path, None) for tag, path in parse_tag_specs(args.dataset_pre, "--dataset_pre")]
    tags = [t for t, _, _ in specs]
    if len(set(tags)) != len(tags):
        raise ValueError(f"TAG duplicati: {tags}")

    all_pairs, dataset_infos = [], []
    for tag, path, dorsal_path in specs:
        print("\n" + "=" * 72)
        if dorsal_path is None:
            print(f"CARICAMENTO DATASET '{tag}' (preprocessed): {path}")
        else:
            print(f"CARICAMENTO DATASET '{tag}' | PALMO: {path} | DORSO: {dorsal_path}")
        print("=" * 72)
        pairs, info = load_preprocessed_dataset(tag, path, dorsal_path)
        print(f"[{tag}] coppie palmo+dorso utilizzabili: {len(pairs)} | "
              f"soggetti: {len({p['raw_subject'] for p in pairs})}")
        all_pairs += pairs
        dataset_infos.append(info)

    # ---- anti-leakage (solo sui dataset da cui provengono i dati di training) ----
    leak_report = None
    if args.leak_policy != "off":
        check_tags = [t.strip() for t in args.leak_check_tags.split(",") if t.strip()]
        bad = set(check_tags) - set(tags)
        if bad:
            raise ValueError(f"--leak_check_tags contiene TAG non caricati: {sorted(bad)}")
        train_ids, eval_ids, log_details = load_training_subjects(args.train_log)
        all_pairs, leak_report = apply_leak_filter(
            all_pairs, train_ids, eval_ids, args.leak_policy, check_tags
        )
        leak_report["training_logs"] = log_details
        save_json(results_dir / "leak_report.json", leak_report)
        print("\n" + "=" * 72)
        print(f"CONTROLLO LEAKAGE (policy={args.leak_policy})")
        print("=" * 72)
        for tag, r in leak_report["per_dataset"].items():
            if r["checked_against_training"]:
                print(f"[{tag}] {r['n_subjects_before']} soggetti | in TRAIN: {r['n_overlap_train']} | "
                      f"in EVAL: {r['n_overlap_eval']} | rimasti: {r['n_subjects_after']}")
            else:
                print(f"[{tag}] {r['n_subjects_before']} soggetti | NON confrontato col training "
                      f"(assunto disgiunto: verifica che non sia stato usato per addestrare)")
    else:
        print("\n[!] --leak_policy off: nessun controllo di leakage.")

    # ---- niente augmentation: escludi le mani con un solo scatto ----
    all_pairs, dropped_hands = drop_single_shot_hands(all_pairs)
    if dropped_hands:
        print(f"\n[i] {len(dropped_hands)} mani escluse perche' hanno un solo scatto reale "
              f"(nessuna augmentation).")

    # ---- dataset con troppi pochi soggetti ----
    kept_tags = []
    for tag in tags:
        n_sub = len({p["subject"] for p in all_pairs if p["dataset"] == tag})
        if n_sub < args.min_subjects_per_dataset:
            print(f"[!] Dataset '{tag}' escluso: {n_sub} soggetti puliti < "
                  f"--min_subjects_per_dataset={args.min_subjects_per_dataset}.")
        else:
            kept_tags.append(tag)
    if not kept_tags:
        raise RuntimeError("Nessun dataset con abbastanza soggetti puliti: vedi leak_report.json.")
    all_pairs = [p for p in all_pairs if p["dataset"] in kept_tags]

    save_json(results_dir / "datasets_info.json",
              {"datasets": dataset_infos, "kept_tags": kept_tags,
               "single_shot_hands_dropped": dropped_hands})

    embeddings = compute_embeddings(
        all_pairs, args.palm_checkpoint, args.dorsal_checkpoint, device=args.device
    )

    scopes = [(f"dataset_{t}", [t]) for t in kept_tags]
    if len(kept_tags) > 1:
        scopes.append(("pooled", list(kept_tags)))

    combined = {}
    for scope, scope_tags in scopes:
        print("\n" + "#" * 72)
        print(f"# SCOPE: {scope}  (dataset: {', '.join(scope_tags)})")
        print("#" * 72)
        emb = [e for e in embeddings if e["dataset"] in scope_tags]
        try:
            summ = run_evaluation(emb, args, results_dir / scope, scope,
                                  {"tags": scope_tags, "leak_report": leak_report})
            combined[scope] = {
                "n_subjects_test": summ["dataset"]["n_subjects"],
                "eer_percent": {v["system"]: v["eer_percent"] for v in summ["verification"]},
                "rank1_percent": {r["system"]: r["rank1_percent"] for r in summ["identification"]},
                "alpha": summ["protocol"]["alpha"],
            }
        except Exception as exc:  # uno scope che fallisce non deve bloccare gli altri
            print(f"[!] Scope '{scope}' fallito: {exc}")
            combined[scope] = {"error": str(exc)}

    save_json(results_dir / "combined_summary.json", combined)
    print("\n" + "=" * 72)
    print("RIEPILOGO (soggetti TEST, nessuna augmentation, nessun leakage)")
    print("=" * 72)
    for scope, c in combined.items():
        if "error" in c:
            print(f"{scope:>16s}: ERRORE - {c['error']}")
            continue
        eer = " ".join(f"{k}={v:.2f}%" for k, v in c["eer_percent"].items())
        r1 = " ".join(f"{k}={v:.2f}%" for k, v in c["rank1_percent"].items())
        print(f"{scope:>16s}: n_sub={c['n_subjects_test']} | EER {eer} | Rank-1 {r1}")
    print(f"\nRisultati completi in: {results_dir}")


def run_evaluation(embeddings, args, results_dir, scope, scope_info):
    """Esegue verifica 1:1, closed-set e open-set su `embeddings` (gia' filtrati per leak)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    leak_report = scope_info.get("leak_report")
    use_score_norm = (args.score_norm == "zscore")

    # DEV (alpha + statistiche z-score) e TEST (numeri finali) su soggetti disgiunti
    dev_info = None
    if args.dev_fraction > 0:
        calib_emb, embeddings, dev_subj, test_subj = split_dev_test(
            embeddings, args.dev_fraction, args.seed
        )
        dev_info = {"dev_fraction": args.dev_fraction, "n_dev_subjects": len(dev_subj),
                    "n_test_subjects": len(test_subj),
                    "dev_subjects": dev_subj, "test_subjects": test_subj}
        save_json(results_dir / "dev_test_split.json", dev_info)
        print(f"\n[SPLIT] DEV (alpha/z-score): {len(dev_subj)} soggetti | "
              f"TEST (risultati finali): {len(test_subj)} soggetti (disgiunti)")
    else:
        calib_emb = embeddings
        print("\n[!] --dev_fraction 0: alpha e z-score calibrati sugli STESSI soggetti del test.")
    subjects = sorted(set(x["subject"] for x in embeddings))
    calib_subjects = sorted(set(x["subject"] for x in calib_emb))
    kfold_eff = max(2, min(args.kfold, len(calib_subjects) // 3))
    rank_kfold_eff = max(2, min(args.rank_kfold, len(calib_subjects) // 3))

    calibration_info = None
    norm_stats = None
    do_cal = args.calibrate_alpha and len(calib_subjects) >= kfold_eff * 3
    do_cal_rank = args.calibrate_alpha_for_rank and len(calib_subjects) >= rank_kfold_eff * 3
    if args.calibrate_alpha and not do_cal:
        print(f"[!] DEV troppo piccolo ({len(calib_subjects)} soggetti) per calibrare alpha: "
              f"uso --alpha {args.alpha}. Riduci --dev_fraction o aggiungi soggetti.")
    if args.calibrate_alpha_for_rank and not do_cal_rank:
        print(f"[!] DEV troppo piccolo per calibrare alpha sul Rank-1: uso l'alpha di verifica.")

    if do_cal:
        calibration_info = calibrate_alpha_kfold(
            calib_emb, calib_subjects, k=kfold_eff, seed=args.seed, use_score_norm=use_score_norm
        )
        alpha = calibration_info["alpha"]

        save_json(results_dir / "alpha_calibration.json", calibration_info)

        print(f"\nAlpha calibrato ({kfold_eff}-fold, subject-disjoint): {alpha:.2f}")
        print(f"EER medio al best alpha: {calibration_info['mean_eer_at_best_alpha']*100:.3f}%")
        print("Curva EER-vs-alpha (controlla se e' un plateau piatto):")
        for a, e in sorted(calibration_info["mean_eer_by_alpha"].items()):
            print(f"  alpha={a:.2f}  EER medio={e*100:.3f}%")

        evaluation_embeddings = embeddings
        if use_score_norm:
            norm_stats = compute_score_stats(calib_emb, seed=args.seed)
    else:
        alpha = float(args.alpha)
        evaluation_embeddings = embeddings
        if use_score_norm:
            norm_stats = compute_score_stats(calib_emb, seed=args.seed)

    print("\n" + "=" * 72)
    print("VERIFICA 1:1")
    print("=" * 72)

    genuine, impostor, verification_protocol = make_verification_pairs(
        evaluation_embeddings,
        max_impostor_pairs=args.max_impostor_pairs,
        seed=args.seed,
    )

    print(f"Protocollo genuine usato: {verification_protocol}")
    print(f"Coppie genuine:  {len(genuine)}")
    print(f"Coppie impostor: {len(impostor)}")
    print(f"Alpha:            {alpha:.2f}")

    scores = pair_scores(evaluation_embeddings, genuine + impostor, alpha, norm_stats=norm_stats)
    n_g = len(genuine)

    systems = {
        "palm": (scores["palm"][:n_g], scores["palm"][n_g:]),
        "dorsal": (scores["dorsal"][:n_g], scores["dorsal"][n_g:]),
        "fused": (scores["fused"][:n_g], scores["fused"][n_g:]),
    }

    verification_results = []
    for name, (g, imp) in systems.items():
        scale_note = None
        if name == "fused" and norm_stats is not None:
            scale_note = (
                "Punteggi z-normalizzati: la soglia fissa non e' direttamente "
                "comparabile con quella di palm/dorsal (scala coseno grezzo). "
                "Usa eer_threshold_metrics per confronti tra sistemi."
            )
        metrics = evaluate_verification(g, imp, fixed_threshold=args.threshold, name=name,
                                          score_scale_note=scale_note)
        metrics["fixed_threshold_metrics"] = metrics.pop("fixed_threshold")
        verification_results.append(metrics)

        print(
            f"{name:>8s}: EER={metrics['eer_percent']:.2f}% | AUC={metrics['roc_auc']:.4f} | "
            f"Acc@EER={metrics['eer_threshold_metrics']['accuracy']*100:.2f}% | "
            f"Acc@{args.threshold:.2f}={metrics['fixed_threshold_metrics']['accuracy']*100:.2f}%"
            + ("  [scala non comparabile, vedi note]" if scale_note else "")
        )

    save_verification_csv(results_dir / "verification_metrics.csv", verification_results)
    save_pair_scores_csv(results_dir / "verification_pairs.csv", evaluation_embeddings,
                          genuine, impostor, alpha, norm_stats=norm_stats)

    print("\n" + "=" * 72)
    print("IDENTIFICAZIONE 1:N")
    print("=" * 72)

    rank_calibration_info = None
    identification_alpha = alpha
    if do_cal_rank:
        rank_calibration_info = calibrate_alpha_for_rank(
            calib_emb, calib_subjects, k=rank_kfold_eff, seed=args.seed,
            gallery_holdout_real=args.gallery_holdout_real,
            norm_stats=norm_stats,
        )
        identification_alpha = rank_calibration_info["alpha"]
        save_json(results_dir / "alpha_calibration_rank.json", rank_calibration_info)

        print(f"\nAlpha calibrato per Rank-1 ({rank_kfold_eff}-fold, subject-disjoint): "
              f"{identification_alpha:.2f}")
        print(f"Rank-1 medio al best alpha: {rank_calibration_info['mean_rank1_at_best_alpha']*100:.2f}%")
        print("Curva Rank-1-vs-alpha:")
        for a, r1 in sorted(rank_calibration_info["mean_rank1_by_alpha"].items()):
            print(f"  alpha={a:.2f}  Rank-1 medio={r1*100:.2f}%")

    identification_metrics, identification_details, identification_protocol = identification_all_systems(
        evaluation_embeddings, identification_alpha, norm_stats=norm_stats,
        gallery_holdout_real=args.gallery_holdout_real,
    )

    print(f"Protocollo identificazione usato: {identification_protocol}")
    print(f"Alpha usato per l'identificazione: {identification_alpha:.2f}")

    for r in identification_metrics:
        if r["gallery_side"] == "Same Hand (Enrollment)":
            extra = ""
            if r["rank1_percent_real_probes"] is not None:
                extra += f" | Rank-1(real)={r['rank1_percent_real_probes']:.2f}%"
            if r["rank1_percent_augmented_probes"] is not None:
                extra += f" | Rank-1(aug)={r['rank1_percent_augmented_probes']:.2f}%"
            print(
                f"{r['system']:>8s}: Rank-1={r['rank1_percent']:.2f}% | "
                f"Rank-5={r['rank5_percent']:.2f}% | Rank-10={r['rank10_percent']:.2f}% | "
                f"MRR={r['mrr']:.4f}{extra}"
            )
            if r["mean_margin_top1_top2"] is not None:
                print(
                    f"           margine medio top1-top2: {r['mean_margin_top1_top2']:.4f} "
                    f"(quando sbagliato: "
                    f"{r['mean_margin_when_wrong']:.4f})" if r["mean_margin_when_wrong"] is not None
                    else f"           margine medio top1-top2: {r['mean_margin_top1_top2']:.4f}"
                )

    save_identification_csv(results_dir / "identification_metrics.csv", identification_metrics)
    save_details_csv(results_dir / "identification_details.csv", identification_details)

    open_set_results = None
    if args.skip_openset:
        print("\nIdentificazione open-set saltata (--skip_openset).")
    else:
        print("\n" + "=" * 72)
        print("IDENTIFICAZIONE OPEN-SET")
        print("=" * 72)
        open_set_results = evaluate_open_set(
            evaluation_embeddings, identification_alpha, norm_stats, args, results_dir
        )

    summary = {
        "protocol": {
            "scope": scope,
            "datasets_in_scope": scope_info["tags"],
            "augmentation": "none (solo scatti reali)",
            "impostors_and_gallery": "solo dello stesso dataset",
            "verification_protocol_used": verification_protocol,
            "identification_protocol_used": identification_protocol,
            "alpha": alpha,
            "alpha_calibrated_kfold": bool(do_cal),
            "kfold": kfold_eff if do_cal else None,
            "identification_alpha": identification_alpha,
            "identification_alpha_calibrated_for_rank": bool(do_cal_rank),
            "rank_kfold": rank_kfold_eff if do_cal_rank else None,
            "gallery_holdout_real": args.gallery_holdout_real,
            "score_normalization": args.score_norm,
            "fixed_verification_threshold": args.threshold,
            "seed": args.seed,
            "leak_policy": args.leak_policy,
            "leak_report": leak_report,
            "dev_test_split": dev_info,
            "open_set_enabled": open_set_results is not None,
            "open_set_repeats": args.openset_repeats if open_set_results is not None else None,
            "open_set_known_fraction": args.openset_known_fraction if open_set_results is not None else None,
            "open_set_unknown_samples": args.openset_unknown_samples if open_set_results is not None else None,
        },
        "dataset": {
            "n_subjects": len(set(x["subject"] for x in embeddings)),
            "n_embedding_samples": len(embeddings),
            "n_subjects_per_dataset_test": {
                tag: len({x["subject"] for x in embeddings if x.get("dataset") == tag})
                for tag in scope_info["tags"]},
        },
        "verification": verification_results,
        "identification": identification_metrics,
        "alpha_calibration": calibration_info,
        "alpha_calibration_rank": rank_calibration_info,
        "open_set": _json_safe(open_set_results) if open_set_results is not None else None,
    }

    save_json(results_dir / "summary.json", summary)

    with open(results_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("RISULTATI MULTI-DATASET [" + scope + "] - PALMO + DORSO (v6)\n")
        f.write("=" * 72 + "\n\n")

        f.write(f"Protocollo verifica:       {verification_protocol}\n")
        f.write(f"Protocollo identificazione: {identification_protocol}\n")
        f.write(f"\nSoggetti utilizzabili: {summary['dataset']['n_subjects']}\n")
        f.write(f"Campioni multimodali: {summary['dataset']['n_embedding_samples']}\n")
        f.write(f"Alpha fusione: {alpha:.2f}\n")
        f.write(f"Soglia verifica fissa: {args.threshold:.4f}\n\n")

        f.write("VERIFICATION 1:1\n" + "-" * 72 + "\n")
        for r in verification_results:
            fixed = r["fixed_threshold_metrics"]
            eer_m = r["eer_threshold_metrics"]
            f.write(
                f"\n{r['system'].upper()}\n"
                f"  EER:                    {r['eer']*100:.4f}%\n"
                f"  EER threshold:          {r['eer_threshold']:.6f}\n"
                f"  ROC-AUC:                {r['roc_auc']:.6f}\n"
                f"  Accuracy @ EER thresh:  {eer_m['accuracy']*100:.4f}%\n"
                f"  Accuracy @ {args.threshold:.2f}: {fixed['accuracy']*100:.4f}%\n"
                f"  TAR @ FAR 1%:           {r['tar_at_far_1pct']['tar']*100:.4f}%\n"
                f"  TAR @ FAR 5%:           {r['tar_at_far_5pct']['tar']*100:.4f}%\n"
                f"  TAR @ FAR 10%:          {r['tar_at_far_10pct']['tar']*100:.4f}%\n"
            )
            if r.get("score_scale_note"):
                f.write(f"  NOTA SCALA: {r['score_scale_note']}\n")

        f.write("\nIDENTIFICATION 1:N\n" + "-" * 72 + "\n")
        for r in identification_metrics:
            if r["gallery_side"] == "Same Hand (Enrollment)":
                f.write(
                    f"\n{r['system'].upper()}\n"
                    f"  Rank-1:  {r['rank1_percent']:.4f}%\n"
                    f"  Rank-5:  {r['rank5_percent']:.4f}%\n"
                    f"  Rank-10: {r['rank10_percent']:.4f}%\n"
                    f"  MRR:     {r['mrr']:.6f}\n"
                )
                if r["rank1_percent_real_probes"] is not None:
                    f.write(f"  Rank-1 (probe reali):      {r['rank1_percent_real_probes']:.4f}%\n")
                if r["rank1_percent_augmented_probes"] is not None:
                    f.write(f"  Rank-1 (probe augmentati): {r['rank1_percent_augmented_probes']:.4f}%\n")
                if r["mean_margin_top1_top2"] is not None:
                    f.write(f"  Margine medio top1-top2:   {r['mean_margin_top1_top2']:.4f}\n")

        if open_set_results is not None:
            f.write("\nIDENTIFICATION OPEN-SET\n" + "-" * 72 + "\n")
            for line in format_open_set_summary(open_set_results):
                f.write(line + "\n")

    print(f"[{scope}] risultati salvati in: {results_dir}")
    return summary


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test multimodale palmo+dorso su piu' dataset - v6, senza augmentation e senza leakage."
    )
    parser.add_argument("--dataset_pre", action="append", default=[], metavar="TAG=DIR",
                        help="Dataset con palmo e dorso nella stessa cartella (ripetibile).")
    parser.add_argument("--palm_pre", action="append", default=[], metavar="TAG=DIR",
                        help="Cartella preprocessata del PALMO; ripetibile, TAG deve corrispondere a --dorsal_pre.")
    parser.add_argument("--dorsal_pre", action="append", default=[], metavar="TAG=DIR",
                        help="Cartella preprocessata del DORSO; ripetibile, TAG deve corrispondere a --palm_pre.")
    parser.add_argument("--palm_checkpoint", default="models_final/palm_embedding_final.pt")
    parser.add_argument("--dorsal_checkpoint", default="models_final_dorsal/dorsal_embedding_final.pt")
    parser.add_argument("--out_dir", default="multi_dataset_results",
                        help="Cartella di output (results/).")

    parser.add_argument("--train_log", nargs="+", default=None,
                        help="Log JSONL di training (palmo e dorso) con i soggetti di train/eval.")
    parser.add_argument("--leak_check_tags", default="",
                        help="TAG (separati da virgola) dei dataset da cui provengono i dati di "
                             "training: solo per questi si escludono i soggetti dei log.")
    parser.add_argument("--leak_policy", choices=["strict", "train_only", "off"], default="strict")
    parser.add_argument("--dev_fraction", type=float, default=0.3,
                        help="Frazione di soggetti per DEV (alpha + z-score), disgiunta dal TEST.")
    parser.add_argument("--min_subjects_per_dataset", type=int, default=8,
                        help="Dataset con meno soggetti puliti vengono esclusi (default: 8).")

    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--score_norm", choices=["none", "zscore"], default="zscore")
    parser.add_argument("--calibrate_alpha", action="store_true")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--calibrate_alpha_for_rank", action="store_true")
    parser.add_argument("--rank_kfold", type=int, default=5)
    parser.add_argument("--gallery_holdout_real", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--skip_openset", action="store_true")
    parser.add_argument("--openset_repeats", type=int, default=10)
    parser.add_argument("--openset_known_fraction", type=float, default=0.5)
    parser.add_argument("--openset_cal_fraction", type=float, default=0.5)
    parser.add_argument("--openset_target_fpirs", default="0.01,0.05,0.10")
    parser.add_argument("--openset_unknown_samples", choices=["real", "all"], default="real")
    parser.add_argument("--max_impostor_pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    if not args.dataset_pre and not (args.palm_pre and args.dorsal_pre):
        parser.error("indica --dataset_pre TAG=DIR oppure entrambe --palm_pre TAG=DIR e --dorsal_pre TAG=DIR.")
    if args.leak_policy != "off":
        if not args.train_log:
            parser.error("--leak_policy strict/train_only richiede --train_log <log palmo> <log dorso>.")
        if not args.leak_check_tags.strip():
            parser.error("indica con --leak_check_tags i TAG dei dataset usati per il training "
                         "(es. --leak_check_tags 11k), oppure usa --leak_policy off.")
    if not 0.0 <= args.dev_fraction < 0.8:
        parser.error("--dev_fraction deve essere in [0, 0.8).")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha deve essere compreso tra 0 e 1.")
    if args.openset_repeats < 1:
        raise ValueError("--openset_repeats deve essere >= 1.")

    evaluate_dataset(args)


if __name__ == "__main__":
    main()