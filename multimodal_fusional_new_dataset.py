"""
MULTIMODAL_FUSION - Valutazione multimodale su nuovo dataset (v3 - niente fallback L/R)
=========================================================================================

Modifiche rispetto alla v2:
- ELIMINATO qualunque fallback L-vs-R (sia in verifica 1:1 che in identificazione 1:N):
  le coppie/gallery-probe genuine sono SEMPRE "stessa mano, scatti diversi". Non esiste
  piu' un modo, nemmeno opzionale, per costruire coppie genuine mescolando mano sx e dx.
- Data augmentation AUTOMATICA per-campione: se una mano ha meno di
  --min_samples_per_hand acquisizioni complete (palmo+dorso), lo script genera in modo
  automatico scatti augmentati di QUELLA STESSA mano (piccola rotazione, jitter di
  luminosita'/contrasto, traslazione) finche' non raggiunge la soglia minima, cosi'
  il protocollo "same_hand_multi_sample" e' sempre applicabile senza dover mai
  ricorrere alla mano opposta.
- --out_dir usato SOLO in scrittura: contiene sempre e solo augmented_raw/,
  preprocessed/ e results/ generati da questa run. La lettura di un preprocessing
  già calcolato in precedenza (--skip_preprocessing) avviene tramite il nuovo
  argomento esplicito --preprocessed_dir, mai in modo implicito da --out_dir.
- Soglia fissa applicata in modo coerente: warning esplicito se un sistema usa punteggi
  z-normalizzati e un altro no.
- Calibrazione alpha via k-fold (invece di un singolo split 20/80), con curva EER-vs-alpha
  salvata per intero, non solo il minimo.
- Ogni file di output (summary.json, summary.txt) riporta il protocollo di verifica e
  quante acquisizioni sono state generate via augmentation.
- Metriche riportate per la verifica 1:1: EER, accuracy alla soglia EER, accuracy alla
  soglia fissa, ROC-AUC, TAR@FAR. Per l'identificazione 1:N: Rank-1, Rank-5, Rank-10, MRR.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

from palm_run import PalmVerifier
from dorsal_run import DorsalVerifier


# ============================================================
# DATASET DISCOVERY
# ============================================================

# Pattern esteso: supporta anche un suffisso _aug<N> per campioni augmentati,
# es. P001_1_L_palmar_aug1_processed.png
IMAGE_RE = re.compile(
    r"^(?P<subject>P\d+)_"
    r"(?:(?P<sess1>s?\d+)_+)?"
    r"(?P<side>L|R)_"
    r"(?:(?P<sess2>s?\d+)_+)?"
    r"(?P<modality>palmar|dorsal)"
    r"(?:_(?P<sess3>s?\d+))?"
    r"(?:_(?P<aug>aug\d+))?"
    r"(?:_processed)?\.(?P<ext>png|jpg|jpeg|bmp)$",
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

        # sample_id combina sessione + tag augmentation, se presente, così
        # ogni combinazione (sessione, augmentation) è un "scatto" distinto
        base_sample = (
            m.group("sess1")
            or m.group("sess2")
            or m.group("sess3")
            or "1"
        )
        aug_tag = m.group("aug")
        sample_id = f"{base_sample}_{aug_tag}" if aug_tag else str(base_sample)

        rec = {
            "subject": m.group("subject").upper(),
            "side": m.group("side").upper(),
            "modality": m.group("modality").lower(),
            "sample_id": sample_id.lower(),
            "is_augmented": bool(aug_tag),
            "path": str(path),
        }
        records.append(rec)

    records.sort(key=lambda x: (x["subject"], x["side"], x["sample_id"], x["modality"]))

    if not records:
        raise RuntimeError(
            f"Nessuna immagine trovata in {root}. "
            "Attesi file nel formato PXXX_L/R_palmar/dorsal_processed.png, "
            "opzionalmente con indicatore di scatto e/o _augN."
        )

    return records


def build_raw_pairs(records):
    by_key = {}
    for r in records:
        key = (r["subject"], r["side"], r["sample_id"])
        by_key.setdefault(key, {})[r["modality"]] = r["path"]

    pairs = []
    missing = []

    for (subject, side, sample_id), modalities in sorted(by_key.items()):
        if "palmar" not in modalities or "dorsal" not in modalities:
            missing.append({
                "subject": subject,
                "side": side,
                "sample_id": sample_id,
                "available": sorted(modalities.keys()),
            })
            continue

        pairs.append({
            "subject": subject,
            "side": side,
            "sample_id": sample_id,
            "palm_raw": modalities["palmar"],
            "dorsal_raw": modalities["dorsal"],
        })

    return pairs, missing


# ============================================================
# AUGMENTATION AUTOMATICA PER-MANO (mai cross-hand: solo sulla STESSA mano)
# ============================================================

def _build_augmented_filename(original_name: str, aug_tag: str) -> str:
    """
    Inserisce il tag di augmentation (es. 'aug1') nel nome file mantenendo il
    formato riconosciuto da IMAGE_RE, cosi' il file augmentato viene
    ri-scoperto come uno scatto (sample_id) distinto della STESSA mano.
    """
    m = IMAGE_RE.match(original_name)
    if not m:
        raise ValueError(f"Nome file non riconosciuto per augmentation: {original_name}")
    ext = m.group("ext")
    lower = original_name.lower()
    processed_suffix = f"_processed.{ext.lower()}"
    if lower.endswith(processed_suffix):
        stem = original_name[: -len(processed_suffix)]
        return f"{stem}_{aug_tag}_processed.{ext}"
    plain_suffix = f".{ext.lower()}"
    stem = original_name[: -len(plain_suffix)]
    return f"{stem}_{aug_tag}.{ext}"


def _detect_background_fill(img: "Image.Image"):
    """
    Stima il colore di sfondo dell'immagine campionando i suoi bordi
    (angoli + meta' dei quattro lati), cosi' rotazione/traslazione possono
    riempire le zone scoperte con un colore coerente con lo sfondo reale
    (es. bianco/grigio chiaro) invece del nero di default di PIL, che altrimenti
    crea cunei/bordi neri innaturali assenti nell'immagine originale.
    """
    rgb_img = img.convert("RGB")
    w, h = rgb_img.size
    px = rgb_img.load()

    sample_points = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2),
    ]
    samples = [px[x, y] for x, y in sample_points]

    r = int(round(sum(s[0] for s in samples) / len(samples)))
    g = int(round(sum(s[1] for s in samples) / len(samples)))
    b = int(round(sum(s[2] for s in samples) / len(samples)))

    if img.mode == "RGB":
        return (r, g, b)
    if img.mode == "RGBA":
        return (r, g, b, 255)
    if img.mode in ("L", "1", "I", "F"):
        # luminosita' media come approssimazione in scala di grigi
        return int(round(0.299 * r + 0.587 * g + 0.114 * b))
    # fallback generico: converti il colore RGB stimato nella modalita' originale
    return Image.new("RGB", (1, 1), (r, g, b)).convert(img.mode).getpixel((0, 0))


def _augment_image_file(src_path: Path, dst_path: Path, rng: random.Random):
    """
    Applica una trasformazione leggera ma non banale (rotazione, luminosita',
    contrasto, piccola traslazione) cosi' da simulare uno 'scatto diverso'
    della stessa mano invece di una copia identica dell'immagine originale.

    Prima di ruotare, aggiunge un margine temporaneo riempito con lo sfondo
    rilevato: se il crop originale e' gia' stretto attorno alla mano, una
    rotazione con expand=False su un canvas SENZA margine taglierebbe pixel
    reali della mano (punte delle dita, polso) ai bordi, ed e' proprio questo
    che fa fallire piu' spesso MediaPipe sugli augmentati rispetto agli
    originali. Il margine da' spazio alla rotazione senza perdere contenuto
    reale; il preprocessing a valle ritaglia comunque la mano in base ai
    landmark rilevati, quindi il margine extra non e' un problema.

    Le zone scoperte da rotazione/traslazione vengono riempite con il colore
    di sfondo rilevato dall'originale (non nero), cosi' l'augmentato ha lo
    stesso sfondo dell'immagine di partenza.
    """
    img = Image.open(src_path)
    orig_mode = img.mode

    fill_color = _detect_background_fill(img)

    # Margine temporaneo: 15% per lato (min 20px), sufficiente per angoli
    # fino a ~8 gradi senza tagliare la mano se il crop originale e' stretto.
    w0, h0 = img.size
    margin_x = max(20, int(w0 * 0.15))
    margin_y = max(20, int(h0 * 0.15))
    padded = Image.new(img.mode, (w0 + 2 * margin_x, h0 + 2 * margin_y), fill_color)
    padded.paste(img, (margin_x, margin_y))
    img = padded

    angle = rng.uniform(-8.0, 8.0)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill_color)

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))

    w, h = img.size
    max_dx = max(1, int(w * 0.03))
    max_dy = max(1, int(h * 0.03))
    dx = rng.randint(-max_dx, max_dx)
    dy = rng.randint(-max_dy, max_dy)
    img = img.transform(
        img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy),
        resample=Image.BICUBIC, fillcolor=fill_color,
    )

    if img.mode != orig_mode:
        img = img.convert(orig_mode)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path)


def ensure_min_samples_per_hand(records, aug_dir: Path, min_samples: int = 2,
                                 n_variants_per_step: int = 1, seed: int = 42):
    """
    Garantisce che ogni mano (subject, side) con una coppia multimodale completa
    (palmo+dorso) abbia almeno `min_samples` scatti, generando augmentazioni
    SOLO a partire da immagini della stessa mano (mai L al posto di R o viceversa).

    I file augmentati vengono scritti in `aug_dir` (mai in data_dir, che resta
    intatto) e poi ri-scoperti con discover_raw_samples, cosi' da riusare la
    stessa logica di parsing/validazione dei file reali.

    `aug_dir` e' una cartella indipendente da --out_dir (tipicamente persistente
    e riusabile tra run diverse), passata esplicitamente dal chiamante: se al
    suo interno esistono gia' scatti augmentati per una mano (da run precedenti),
    NON vengono rigenerati ne' duplicati, ma riusati cosi' come sono.

    Ritorna (records_combinati, report) dove report elenca, per ogni mano
    augmentata, quanti scatti sono stati aggiunti in QUESTA run.
    """
    rng = random.Random(seed)

    by_key = {}
    for r in records:
        key = (r["subject"], r["side"], r["sample_id"])
        by_key.setdefault(key, {})[r["modality"]] = r

    complete_by_hand = {}
    for (subject, side, sample_id), mods in by_key.items():
        if "palmar" in mods and "dorsal" in mods:
            complete_by_hand.setdefault((subject, side), []).append(sample_id)

    # Scatti augmentati gia' presenti in aug_dir da run precedenti (se aug_dir
    # esiste gia'): li riusiamo, non li rigeneriamo mai.
    existing_aug_records = []
    if aug_dir.exists():
        try:
            existing_aug_records = [
                r for r in discover_raw_samples(str(aug_dir)) if r["is_augmented"]
            ]
        except RuntimeError:
            existing_aug_records = []  # aug_dir esiste ma e' vuota

    existing_aug_count = {}
    for r in existing_aug_records:
        existing_aug_count[(r["subject"], r["side"])] = (
            existing_aug_count.get((r["subject"], r["side"]), 0) + 1
        )

    hands_augmented = []
    any_generated = False

    for (subject, side), sample_ids in sorted(complete_by_hand.items()):
        n_have_real = len(sample_ids)
        n_have_aug = existing_aug_count.get((subject, side), 0)
        n_needed = max(0, min_samples - n_have_real - n_have_aug)
        if n_needed == 0:
            continue

        base_sample_id = sorted(sample_ids)[0]
        base_mods = by_key[(subject, side, base_sample_id)]
        generated_here = 0
        next_aug_idx = n_have_aug + 1

        for k in range(next_aug_idx, next_aug_idx + n_needed):
            aug_tag = f"aug{k}"
            for modality, rec in base_mods.items():
                src_path = Path(rec["path"])
                new_name = _build_augmented_filename(src_path.name, aug_tag)
                dst_path = aug_dir / subject / new_name
                _augment_image_file(src_path, dst_path, rng)
            generated_here += 1
            any_generated = True

        hands_augmented.append({
            "subject": subject, "side": side,
            "real_samples": n_have_real,
            "reused_generated_samples": n_have_aug,
            "new_generated_samples": generated_here,
        })

    if not any_generated and not existing_aug_records:
        return records, {"hands_augmented": [], "n_new_raw_images": 0}

    augmented_records = discover_raw_samples(str(aug_dir))
    combined = records + augmented_records

    return combined, {
        "hands_augmented": hands_augmented,
        "n_new_raw_images": len(augmented_records),
    }


# ============================================================
# PREPROCESSING (invariato rispetto alla v1)
# ============================================================

def _is_already_preprocessed(rec, output_dir: Path) -> bool:
    """
    True se per questo singolo file raw (subject/side/sample_id/modality)
    esiste gia' un output di preprocessing in output_dir (un'immagine gia'
    generata in una run precedente, tipicamente un originale o un augmentato
    riusato da aug_dir). Usato in modalita' incrementale per non rielaborare
    ne' duplicare cio' che c'e' gia'.
    """
    subject_dir = output_dir / rec["subject"]
    if not subject_dir.exists():
        return False
    view_word = rec["modality"]
    anchor_suffix = "palm_roi.png" if view_word == "palmar" else "dorsal_hand.png"
    # match ampio: side + view_word + sample_id (se non "1") nel nome file
    for f in subject_dir.glob(f"*{rec['side']}*{view_word}*{anchor_suffix}"):
        if rec["sample_id"] == "1" or rec["sample_id"].lower() in f.name.lower():
            return True
    return False


def preprocess_new_dataset(records, output_dir: Path, incremental: bool = False):
    try:
        from preprocessing_new_dataset import init_worker, process_single_image
    except ImportError as exc:
        raise ImportError(
            "Impossibile importare preProcessing.py. "
            "Assicurati che lo script sia nella root del repository "
            "e che MediaPipe sia installato."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    init_worker()

    if incremental:
        todo = [r for r in records if not _is_already_preprocessed(r, output_dir)]
        n_reused = len(records) - len(todo)
        print(
            f"[PREPROCESS incrementale] {n_reused} file gia' presenti in "
            f"{output_dir} (riusati, non rielaborati), {len(todo)} da processare."
        )
        records = todo

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

        # Prima era hardcoded a 1: originale e augmentati (aug1, aug2, ...)
        # dello stesso subject/side/modality finivano con lo stesso base_name
        # e si sovrascrivevano a vicenda in output_dir. Passando sample_id
        # (che include gia' il tag "aug<N>" quando presente, vedi
        # discover_raw_samples) ogni scatto ottiene un nome distinto.
        task = (rec["path"], rec["subject"], hand_side, is_dorsal, rec["sample_id"], str(output_dir))

        result = process_single_image(task)
        result["subject"] = rec["subject"]
        result["side"] = rec["side"]
        result["sample_id"] = rec["sample_id"]
        result["modality"] = rec["modality"]
        report["results"].append(result)
        print(f"[PREPROCESS] {rec['subject']} {rec['side']} s{rec['sample_id']} {rec['modality']}: {result['status']}")
        print(result)

        if result["status"] == "success":
            report["success"] += 1
        elif result["status"] == "skipped":
            report["skipped"] += 1
        else:
            report["errors"] += 1

        print(
            f"[PREPROCESS] {rec['subject']} {rec['side']} s{rec['sample_id']} "
            f"{rec['modality']}: {result['status']}"
        )

    report_name = "preprocessing_report_incremental.json" if incremental else "preprocessing_report.json"
    report_path = (output_dir if incremental else output_dir.parent) / report_name
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


# ============================================================
# PROCESSED DATASET DISCOVERY (invariato)
# ============================================================

def find_processed_base(preprocessed_dir: Path, subject: str, side: str, sample_id: str):
    subject_dir = preprocessed_dir / subject

    def find_unique_base(view_word: str, anchor_suffix: str):
        patterns = [
            f"{subject}_*_{side}_*_{view_word}_*_{anchor_suffix}.png",
            f"{subject}_{side}_*_{view_word}_*_{anchor_suffix}.png",
            f"{subject}_*_{view_word}_*_{anchor_suffix}.png",
        ]
        matches = []
        for pat in patterns:
            matches = sorted(subject_dir.glob(pat))
            if matches:
                break

        if not matches:
            matches = sorted(subject_dir.glob(f"*{side}*{view_word}*_{anchor_suffix}.png"))

        if not matches:
            raise FileNotFoundError(
                f"Nessun file trovato per {subject} {side} ({view_word}) in {subject_dir}"
            )

        if len(matches) > 1 and sample_id != "1":
            filtered = [m for m in matches if sample_id.lower() in m.name.lower()]
            if len(filtered) == 1:
                matches = filtered

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
        "sample_id": sample_id,
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
                    p["sample_id"],
                )
            )
        except Exception as exc:
            failures.append({
                "subject": p["subject"],
                "side": p["side"],
                "sample_id": p["sample_id"],
                "error": str(exc),
            })

    return pairs, failures


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
            "almeno 2 acquisizioni (reali o augmentate) per mano. Aumenta "
            "--min_samples_per_hand o verifica che l'augmentation automatica "
            "sia attiva (non usare --skip_preprocessing su dati mai augmentati)."
        )

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if embeddings[i]["subject"] != embeddings[j]["subject"]:
                all_impostor.append((i, j))

    rng.shuffle(all_impostor)

    n_imp = len(genuine)
    if max_impostor_pairs is not None:
        n_imp = min(n_imp, int(max_impostor_pairs))

    impostor = all_impostor[:n_imp]

    if len(genuine) > len(impostor) and len(impostor) > 0:
        genuine = genuine[:len(impostor)]

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

    fars = np.array([np.mean(impostor >= t) for t in thresholds])
    frrs = np.array([np.mean(genuine < t) for t in thresholds])

    idx = int(np.argmin(np.abs(fars - frrs)))
    threshold = float(thresholds[idx])
    eer = float((fars[idx] + frrs[idx]) / 2.0)

    return {
        "eer": eer, "eer_threshold": threshold,
        "eer_far": float(fars[idx]), "eer_frr": float(frrs[idx]),
    }


def tar_at_far(genuine, impostor, target_far):
    impostor = np.asarray(impostor, dtype=np.float64)
    genuine = np.asarray(genuine, dtype=np.float64)

    if len(impostor) == 0:
        return {"target_far": target_far, "threshold": float("nan"),
                "actual_far": float("nan"), "tar": float("nan")}

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
        return {"target_far": target_far, "threshold": t,
                "actual_far": float(np.mean(impostor >= t)),
                "tar": float(np.mean(genuine >= t))}

    tar, threshold, actual_far = max(valid, key=lambda x: x[0])
    return {"target_far": target_far, "threshold": float(threshold),
            "actual_far": float(actual_far), "tar": float(tar)}


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


def pair_scores(embeddings, pairs, alpha, norm_stats=None):
    palm, dorsal, fused = [], [], []

    for i, j in pairs:
        sp = cosine(embeddings[i]["palm_embedding"], embeddings[j]["palm_embedding"])
        sd = cosine(embeddings[i]["dorsal_embedding"], embeddings[j]["dorsal_embedding"])

        if norm_stats is not None:
            sp_fusion = _normalize(sp, norm_stats["palm"])
            sd_fusion = _normalize(sd, norm_stats["dorsal"])
        else:
            sp_fusion, sd_fusion = sp, sd

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

def calibrate_alpha_for_rank(embeddings, subjects, k=5, seed=42, gallery_holdout_real=1):
    """
    L'alpha che minimizza l'EER in verifica 1:1 non e' detto sia l'alpha che
    massimizza il Rank-1 in identificazione 1:N: la verifica separa due
    distribuzioni, l'identificazione deve invece ordinare correttamente N
    candidati. Questa funzione fa una grid search subject-disjoint su alpha
    massimizzando il Rank-1 medio (fused) sui k fold.
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
                gallery_map, probes, float(alpha), "fused_calib", norm_stats=None
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
            "Identificazione 1:N: nessuna mano con almeno 2 scatti disponibile. "
            "Non esiste un fallback cross-hand (L/R): aumenta "
            "--min_samples_per_hand cosi' l'augmentation automatica generi "
            "abbastanza scatti per ogni mano."
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

def evaluate_dataset(args):
    # --out_dir e' usato SOLO in scrittura: qui dentro la run corrente crea
    # (e sola lei) augmented_raw/, preprocessed/ e results/. Non viene mai
    # letto come sorgente implicita di input: il dataset raw si legge sempre
    # e solo da --data_dir, e un preprocessing gia' calcolato in precedenza
    # si legge sempre e solo da --preprocessed_dir (esplicito).
    out_root = Path(args.out_dir)
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Cartella persistente e indipendente da --out_dir dove salvare (e riusare
    # tra run diverse) i RAW augmentati. Default: out_dir/augmented_raw, ma
    # tipicamente la si punta a una cartella stabile fuori da out_dir.
    aug_dir = Path(args.augmented_raw_dir) if args.augmented_raw_dir else out_root / "augmented_raw"

    incremental_preprocessing = False
    if args.skip_preprocessing:
        if args.preprocessed_dir is None:
            raise ValueError(
                "--skip_preprocessing richiede --preprocessed_dir esplicito "
                "(la cartella con il preprocessing gia' calcolato in una run "
                "precedente). --out_dir non viene mai riusato come input."
            )
        preprocessed_dir = Path(args.preprocessed_dir)
        if not preprocessed_dir.exists():
            raise FileNotFoundError(f"--preprocessed_dir non trovato: {preprocessed_dir}")
    elif args.preprocessed_dir is not None:
        # Modalita' incrementale: --preprocessed_dir punta a una cartella GIA'
        # esistente (es. quella con gli originali preprocessati). Lo script
        # preprocessa SOLO i campioni non ancora presenti (tipicamente i nuovi
        # augmentati) e li aggiunge li', senza toccare/cancellare il resto.
        preprocessed_dir = Path(args.preprocessed_dir)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        incremental_preprocessing = True
    else:
        preprocessed_dir = out_root / "preprocessed"

    records = discover_raw_samples(args.data_dir)
    raw_pairs, missing_pairs = build_raw_pairs(records)

    print("\n" + "=" * 72)
    print("NUOVO DATASET - EVALUATION (v3)")
    print("=" * 72)
    print(f"Immagini reali trovate:  {len(records)}")
    print(f"Coppie multimodali reali:{len(raw_pairs)}")
    print(f"Coppie incomplete:       {len(missing_pairs)}")
    print(f"Soggetti:                {len(set(x['subject'] for x in raw_pairs))}")

    # Augmentation automatica SOLO sulla stessa mano: se una mano ha meno di
    # --min_samples_per_hand scatti completi, genera scatti augmentati di
    # quella stessa mano (mai dell'altra) finche' non raggiunge la soglia,
    # cosi' il protocollo di verifica/identificazione "stessa mano, scatti
    # diversi" e' sempre applicabile, senza alcun fallback L-vs-R.
    records, aug_report = ensure_min_samples_per_hand(
        records, aug_dir, min_samples=args.min_samples_per_hand, seed=args.seed,
    )
    raw_pairs, missing_pairs = build_raw_pairs(records)
    n_augmented = sum(1 for r in records if r.get("is_augmented"))

    if aug_report["hands_augmented"]:
        print("\n" + "=" * 72)
        print("AUGMENTATION AUTOMATICA (per-mano, mai cross-hand)")
        print("=" * 72)
        for h in aug_report["hands_augmented"]:
            print(
                f"  {h['subject']} {h['side']}: {h['real_samples']} scatti reali, "
                f"{h['reused_generated_samples']} augmentati riusati "
                f"-> +{h['new_generated_samples']} nuovi augmentati"
            )
        print(f"Immagini raw augmentate totali disponibili: {aug_report['n_new_raw_images']} "
              f"(cartella persistente: {aug_dir})")

    print(f"\nImmagini totali (reali+augmentate): {len(records)}  "
          f"(di cui augmentate: {n_augmented})")
    print(f"Coppie multimodali totali:           {len(raw_pairs)}")

    if args.skip_preprocessing:
        print("\nPreprocessing saltato (--skip_preprocessing).")
    else:
        print("\n" + "=" * 72)
        print("PREPROCESSING")
        print("=" * 72)

        if incremental_preprocessing and args.clean_preprocessed:
            raise ValueError(
                "--clean_preprocessed non e' compatibile con la modalita' "
                "incrementale (--preprocessed_dir senza --skip_preprocessing): "
                "cancellerebbe anche il preprocessing degli originali gia' "
                "esistente in quella cartella."
            )

        if preprocessed_dir.exists() and args.clean_preprocessed:
            shutil.rmtree(preprocessed_dir)

        preprocessing_report = preprocess_new_dataset(
            records, preprocessed_dir, incremental=incremental_preprocessing,
        )
        print(
            f"\nPreprocessing: success={preprocessing_report['success']} "
            f"skipped={preprocessing_report['skipped']} "
            f"errors={preprocessing_report['errors']}"
        )

    processed_pairs, preprocess_failures = build_processed_pairs(preprocessed_dir, raw_pairs)
    print(f"\nCampioni multimodali utilizzabili: {len(processed_pairs)}")

    if len(processed_pairs) < 4:
        raise RuntimeError("Troppi pochi campioni dopo il preprocessing.")

    embeddings = compute_embeddings(
        processed_pairs, args.palm_checkpoint, args.dorsal_checkpoint, device=args.device
    )

    subjects = sorted(set(x["subject"] for x in embeddings))
    use_score_norm = (args.score_norm == "zscore")

    calibration_info = None
    norm_stats = None

    if args.calibrate_alpha:
        calibration_info = calibrate_alpha_kfold(
            embeddings, subjects, k=args.kfold, seed=args.seed, use_score_norm=use_score_norm
        )
        alpha = calibration_info["alpha"]

        save_json(results_dir / "alpha_calibration.json", calibration_info)

        print(f"\nAlpha calibrato ({args.kfold}-fold, subject-disjoint): {alpha:.2f}")
        print(f"EER medio al best alpha: {calibration_info['mean_eer_at_best_alpha']*100:.3f}%")
        print("Curva EER-vs-alpha (controlla se e' un plateau piatto):")
        for a, e in sorted(calibration_info["mean_eer_by_alpha"].items()):
            print(f"  alpha={a:.2f}  EER medio={e*100:.3f}%")

        evaluation_embeddings = embeddings
        if use_score_norm:
            norm_stats = compute_score_stats(evaluation_embeddings, seed=args.seed)
    else:
        alpha = float(args.alpha)
        evaluation_embeddings = embeddings
        if use_score_norm:
            norm_stats = compute_score_stats(evaluation_embeddings, seed=args.seed)

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
    if args.calibrate_alpha_for_rank:
        rank_calibration_info = calibrate_alpha_for_rank(
            evaluation_embeddings, subjects, k=args.rank_kfold, seed=args.seed,
            gallery_holdout_real=args.gallery_holdout_real,
        )
        identification_alpha = rank_calibration_info["alpha"]
        save_json(results_dir / "alpha_calibration_rank.json", rank_calibration_info)

        print(f"\nAlpha calibrato per Rank-1 ({args.rank_kfold}-fold, subject-disjoint): "
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

    summary = {
        "protocol": {
            "dataset_type": "external_new_dataset",
            "input_format": "PXXX_[SESS]_L/R_palmar/dorsal[_augN]_processed.png",
            "verification_protocol_used": verification_protocol,
            "identification_protocol_used": identification_protocol,
            "min_samples_per_hand": args.min_samples_per_hand,
            "hands_auto_augmented": aug_report["hands_augmented"],
            "alpha": alpha,
            "alpha_calibrated_kfold": bool(args.calibrate_alpha),
            "kfold": args.kfold if args.calibrate_alpha else None,
            "identification_alpha": identification_alpha,
            "identification_alpha_calibrated_for_rank": bool(args.calibrate_alpha_for_rank),
            "rank_kfold": args.rank_kfold if args.calibrate_alpha_for_rank else None,
            "gallery_holdout_real": args.gallery_holdout_real,
            "score_normalization": args.score_norm,
            "fixed_verification_threshold": args.threshold,
            "seed": args.seed,
        },
        "dataset": {
            "input_images": len(records),
            "input_images_augmented": n_augmented,
            "multimodal_pairs_found": len(raw_pairs),
            "multimodal_pairs_usable": len(processed_pairs),
            "n_subjects": len(set(x["subject"] for x in embeddings)),
            "n_embedding_samples": len(embeddings),
            "missing_pairs": missing_pairs,
            "preprocess_failures": preprocess_failures,
        },
        "verification": verification_results,
        "identification": identification_metrics,
        "alpha_calibration": calibration_info,
        "alpha_calibration_rank": rank_calibration_info,
    }

    save_json(results_dir / "summary.json", summary)

    with open(results_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("RISULTATI NUOVO DATASET - PALMO + DORSO (v3)\n")
        f.write("=" * 72 + "\n\n")

        f.write(f"Protocollo verifica:       {verification_protocol}\n")
        f.write(f"Protocollo identificazione: {identification_protocol}\n")
        if summary["protocol"]["hands_auto_augmented"]:
            f.write(
                f"\n[i] Mani con augmentation automatica: "
                f"{len(summary['protocol']['hands_auto_augmented'])} "
                f"(vedi 'hands_auto_augmented' in summary.json)\n"
            )
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

    print("\n" + "=" * 72)
    print("TEST COMPLETATO")
    print("=" * 72)
    print(f"Risultati salvati in: {results_dir}")
    if summary["protocol"]["hands_auto_augmented"]:
        print(
            f"\n[i] {len(summary['protocol']['hands_auto_augmented'])} mani "
            "hanno ricevuto augmentation automatica per raggiungere "
            f"--min_samples_per_hand={args.min_samples_per_hand}."
        )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test multimodale palmo+dorso - v3, niente fallback L/R."
    )

    parser.add_argument("--data_dir", required=True,
                         help="Cartella con il dataset RAW di input (sola lettura).")
    parser.add_argument("--palm_checkpoint", default="models_final/palm_embedding_final.pt")
    parser.add_argument("--dorsal_checkpoint", default="models_final_dorsal/dorsal_embedding_final.pt")
    parser.add_argument(
        "--out_dir", default="new_dataset_preprocessed",
        help="Cartella usata SOLO in scrittura per augmented_raw/, preprocessed/ "
             "e results/ di questa run. Non viene mai letta come input implicito.",
    )
    parser.add_argument(
        "--preprocessed_dir", default=None,
        help="Cartella con il preprocessing gia' calcolato. Due usi: "
             "(1) insieme a --skip_preprocessing: usata SOLO in lettura, "
             "nessun nuovo preprocessing viene fatto; "
             "(2) SENZA --skip_preprocessing: modalita' incrementale, "
             "vengono preprocessati solo i campioni non ancora presenti "
             "(tipicamente i nuovi augmentati) e aggiunti li' dentro, senza "
             "toccare/cancellare cio' che gia' c'e' (es. gli originali).",
    )
    parser.add_argument(
        "--augmented_raw_dir", default=None,
        help="Cartella persistente e indipendente da --out_dir dove salvare "
             "(e riusare tra run diverse) i RAW augmentati generati "
             "automaticamente. Se non specificata: <out_dir>/augmented_raw. "
             "Punta sempre alla STESSA cartella tra una run e l'altra per "
             "evitare di rigenerare augmentazioni gia' fatte.",
    )
    parser.add_argument(
        "--min_samples_per_hand", type=int, default=2,
        help="Numero minimo di scatti (reali + augmentati) richiesti per ogni "
             "mano. Se una mano ne ha meno, vengono generati automaticamente "
             "scatti augmentati DI QUELLA STESSA MANO (mai dell'altra) fino a "
             "raggiungere questa soglia. Default: 2 (minimo per una coppia "
             "genuine di verifica).",
    )
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--score_norm", choices=["none", "zscore"], default="zscore")
    parser.add_argument("--calibrate_alpha", action="store_true")
    parser.add_argument("--kfold", type=int, default=5,
                         help="Numero di fold per la calibrazione alpha su EER/verifica (default: 5).")
    parser.add_argument(
        "--calibrate_alpha_for_rank", action="store_true",
        help="Se attivo, calibra un alpha DEDICATO all'identificazione 1:N "
             "massimizzando il Rank-1 medio (k-fold subject-disjoint), invece "
             "di riusare l'alpha calibrato/fissato per la verifica 1:1.",
    )
    parser.add_argument("--rank_kfold", type=int, default=5,
                         help="Numero di fold per la calibrazione alpha su Rank-1 (default: 5).")
    parser.add_argument(
        "--gallery_holdout_real", type=int, default=1,
        help="Numero di scatti REALI per mano da tenere fuori dalla gallery "
             "(usati come probe) quando ce ne sono abbastanza. La gallery e' "
             "la media degli scatti reali di enrollment restanti. Default: 1.",
    )
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--max_impostor_pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_preprocessing", action="store_true")
    parser.add_argument("--clean_preprocessed", action="store_true")

    args = parser.parse_args()

    if args.min_samples_per_hand < 2:
        raise ValueError(
            "--min_samples_per_hand deve essere >= 2: senza almeno 2 scatti "
            "per mano non esiste alcuna coppia genuine valida (e non c'e' "
            "alcun fallback L-vs-R a cui ricorrere)."
        )

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha deve essere compreso tra 0 e 1.")

    evaluate_dataset(args)


if __name__ == "__main__":
    main()