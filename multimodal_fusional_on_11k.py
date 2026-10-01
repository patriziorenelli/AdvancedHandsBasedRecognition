"""
MULTIMODAL_FUSION - Valutazione multimodale su nuovo dataset (v4 - + identificazione open-set)
USATO NEW_DATASET PER TRAINING MODELLI E 11K PER TESTING
=========================================================================================

Fix v4.3 (PERFORMANCE su dataset grandi, es. 11k Hands): la verifica 1:1 era
  quadratica (tutte le coppie impostor in liste Python, coseni calcolati uno a uno con
  torch, EER/TAR@FAR con un np.mean per ogni soglia). Ora: coppie impostor CAMPIONATE
  (default 500k, --max_impostor_pairs), coseni calcolati a blocchi con numpy, EER e
  TAR@FAR via ordinamento + searchsorted, identificazione 1:N vettorizzata.

Fix v4.1: calibrate_alpha_for_rank usa ora gli stessi norm_stats (z-score) della
  valutazione finale, cosi' alpha e' scelto e applicato nella stessa scala.

Fix v4.2: nuove opzioni --preprocessed_only (usa direttamente la cartella del
  preprocessing gia' calcolato, senza raw, preprocessing ne' augmentation) e
  --no_augmentation (disattiva l'augmentation automatica nella modalita' con raw).

Novita' v4:
- IDENTIFICAZIONE OPEN-SET (verifica 1:1 e closed-set 1:N restano invariate): i soggetti
  del nuovo dataset vengono divisi in KNOWN (iscritti in gallery) e UNKNOWN (mai iscritti,
  a loro volta divisi in unknown_cal per scegliere la soglia e unknown_test per misurarla).
  Metriche: FPIR, FNIR, DIR@1/@5, FRR dei known, misidentificazioni, open-set EER, AUC
  known-vs-unknown. Split casuali subject-disjoint ripetuti (--openset_repeats), risultati
  come media±std. Disattivabile con --skip_openset.
- Domain shift: i checkpoint sono addestrati su un dataset diverso, quindi le soglie del
  training non sono trasferibili. La soglia open-set viene calibrata sul NUOVO dataset,
  su soggetti unknown disgiunti da quelli di test.
- Nuovi output in results/: open_set_metrics.json, open_set_curve_split0.csv,
  open_set_details_split0.csv (+ sezione OPEN-SET in summary.txt/summary.json).

Modifiche rispetto alla v2 (v3):
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


def discover_preprocessed_samples(preprocessed_dir: Path):
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
    root = Path(preprocessed_dir)
    groups = {}
    unparsed = []
    seen = set()
    duplicates = []

    for path in sorted(root.rglob("*.png")):
        anchor = next((a for a in _PRE_ANCHORS if path.name.endswith(a)), None)
        if anchor is None:
            continue
        modality = _PRE_ANCHORS[anchor]
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
            f"Nessun campione preprocessato utilizzabile in {root}. Attesi file "
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
# EMBEDDING (invariato)
# ============================================================

def compute_embeddings(pairs, palm_ckpt, dorsal_ckpt, device=None):
    print("\nCaricamento PalmVerifier...")
    palm = PalmVerifier(palm_ckpt, device=device)

    print("\nCaricamento DorsalVerifier...")
    dorsal = DorsalVerifier(dorsal_ckpt, device=device)

    def _cpu(x):
        return x.detach().cpu() if isinstance(x, torch.Tensor) else x

    embeddings = []
    n = len(pairs)

    with torch.no_grad():
        for idx, p in enumerate(pairs, start=1):
            if idx == 1 or idx % 50 == 0 or idx == n:
                print(f"[EMBED] {idx}/{n} {p['subject']} {p['side']} s{p['sample_id']}")

            p_emb = _cpu(palm.embed(p["palm_base"]))
            d_emb = _cpu(dorsal.embed(p["dorsal_base"]))

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


DEFAULT_MAX_IMPOSTOR_PAIRS = 500_000
_STACK_CACHE = {"ref": None, "n": -1, "P": None, "D": None}


def _emb_to_np(t):
    """Embedding -> vettore numpy float64 L2-normalizzato."""
    if isinstance(t, torch.Tensor):
        v = t.detach().cpu().float().numpy().astype(np.float64).reshape(-1)
    else:
        v = np.asarray(t, dtype=np.float64).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _get_unit_stacks(embeddings):
    """Matrici [N, d] di embedding palmo/dorso normalizzati (cache a 1 slot)."""
    c = _STACK_CACHE
    if c["ref"] is embeddings and c["n"] == len(embeddings):
        return c["P"], c["D"]
    P = np.stack([_emb_to_np(e["palm_embedding"]) for e in embeddings])
    D = np.stack([_emb_to_np(e["dorsal_embedding"]) for e in embeddings])
    c.update({"ref": embeddings, "n": len(embeddings), "P": P, "D": D})
    return P, D


# ============================================================
# VERIFICATION PAIRS - PROTOCOLLO CORRETTO, NIENTE FALLBACK SILENZIOSO
# ============================================================

def make_verification_pairs(embeddings, max_impostor_pairs=None, seed=42):
    """
    Ritorna (genuine, impostor, protocol_used).

    Genuine = stessa mano (stesso subject E side), scatti diversi (tutte le
    coppie disponibili, raggruppando per mano: nessun doppio ciclo su N).

    Impostor = soggetti diversi. NON si enumerano mai tutte le N^2/2 coppie
    (con 11k Hands sono decine di milioni): se le coppie possibili sono poche
    (N <= ~2800) si prendono tutte (o un sottoinsieme casuale di dimensione
    max_impostor_pairs), altrimenti si campionano a caso, senza duplicati.
    Default: DEFAULT_MAX_IMPOSTOR_PAIRS (500k), piu' che sufficiente per EER,
    AUC e TAR@FAR=1% stabili.
    """
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    n_imp = int(max_impostor_pairs) if max_impostor_pairs else DEFAULT_MAX_IMPOSTOR_PAIRS

    hand_groups = {}
    for idx, e in enumerate(embeddings):
        hand_groups.setdefault((e["subject"], e["side"]), []).append(idx)

    genuine = []
    for ids in hand_groups.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                genuine.append((ids[a], ids[b]))

    protocol_used = "same_hand_multi_sample"

    if not genuine:
        raise RuntimeError(
            "Nessuna coppia 'stessa mano, scatti diversi' disponibile: servono "
            "almeno 2 acquisizioni (reali o augmentate) per mano. Aumenta "
            "--min_samples_per_hand o verifica che l'augmentation automatica "
            "sia attiva (non usare --skip_preprocessing su dati mai augmentati)."
        )

    _, subj = np.unique(np.array([e["subject"] for e in embeddings]), return_inverse=True)
    subj = subj.astype(np.int64)

    if n * (n - 1) // 2 <= 4_000_000:
        iu, ju = np.triu_indices(n, k=1)
        m = subj[iu] != subj[ju]
        iu, ju = iu[m], ju[m]
        if len(iu) > n_imp:
            sel = rng.choice(len(iu), size=n_imp, replace=False)
            iu, ju = iu[sel], ju[sel]
    else:
        keys = np.empty(0, dtype=np.int64)
        for _ in range(10):
            missing = n_imp - len(keys)
            if missing <= 0:
                break
            i = rng.integers(0, n, size=int(missing * 1.5) + 1000)
            j = rng.integers(0, n, size=int(missing * 1.5) + 1000)
            m = subj[i] != subj[j]
            i, j = i[m], j[m]
            a = np.minimum(i, j)
            b = np.maximum(i, j)
            keys = np.unique(np.concatenate([keys, a * n + b]))
        keys = rng.permutation(keys)[:n_imp]
        iu, ju = keys // n, keys % n

    impostor = list(zip(iu.tolist(), ju.tolist()))
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
    """EER via ordinamento + searchsorted: O((G+I) log(G+I)) invece di O(T*I)."""
    g = np.sort(np.asarray(genuine, dtype=np.float64))
    im = np.sort(np.asarray(impostor, dtype=np.float64))

    scores = np.unique(np.concatenate([g, im]))
    if len(scores) == 1:
        thresholds = scores
    else:
        thresholds = np.concatenate([
            [scores[0] - 1e-7],
            (scores[:-1] + scores[1:]) / 2.0,
            [scores[-1] + 1e-7],
        ])

    fars = 1.0 - np.searchsorted(im, thresholds, side="left") / len(im)
    frrs = np.searchsorted(g, thresholds, side="left") / len(g)

    idx = int(np.argmin(np.abs(fars - frrs)))
    return {
        "eer": float((fars[idx] + frrs[idx]) / 2.0),
        "eer_threshold": float(thresholds[idx]),
        "eer_far": float(fars[idx]), "eer_frr": float(frrs[idx]),
    }


def tar_at_far(genuine, impostor, target_far):
    """TAR alla soglia piu' bassa con FAR <= target_far (ordinamento, O(I log I))."""
    impostor = np.asarray(impostor, dtype=np.float64)
    genuine = np.asarray(genuine, dtype=np.float64)

    if len(impostor) == 0:
        return {"target_far": target_far, "threshold": float("nan"),
                "actual_far": float("nan"), "tar": float("nan")}

    t = threshold_at_fpir(impostor, target_far)
    return {"target_far": target_far, "threshold": float(t),
            "actual_far": float(np.mean(impostor >= t)),
            "tar": float(np.mean(genuine >= t))}


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


def pair_scores(embeddings, pairs, alpha, norm_stats=None, chunk=20000):
    """Punteggi coseno palmo/dorso/fusi per una lista di coppie (a blocchi, numpy)."""
    P, D = _get_unit_stacks(embeddings)
    ij = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    sp = np.empty(len(ij), dtype=np.float64)
    sd = np.empty(len(ij), dtype=np.float64)
    for s in range(0, len(ij), chunk):
        a = ij[s:s + chunk, 0]
        b = ij[s:s + chunk, 1]
        sp[s:s + chunk] = np.einsum("ij,ij->i", P[a], P[b])
        sd[s:s + chunk] = np.einsum("ij,ij->i", D[a], D[b])

    if norm_stats is not None:
        spf = _normalize(sp, norm_stats["palm"])
        sdf = _normalize(sd, norm_stats["dorsal"])
    else:
        spf, sdf = sp, sd

    return {"palm": sp, "dorsal": sd, "fused": alpha * spf + (1.0 - alpha) * sdf}


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
                              debug_tie_threshold=1e-6, verbose_debug=False, chunk=1000):
    """
    Identificazione 1:N vettorizzata: i punteggi probe x gallery si calcolano a
    blocchi con prodotti matriciali (nessun coseno torch per singola coppia).
    Il probe viene confrontato SOLO con le identita' della stessa mano (side).
    """
    details = []
    rank1 = rank5 = rank10 = 0
    reciprocal_sum = 0.0
    n_tied_top1 = 0

    gal_labels = list(gallery_map.keys())
    gal_sides = np.array([l.rsplit("_", 1)[-1] for l in gal_labels])
    label_index = {l: i for i, l in enumerate(gal_labels)}

    if probes:
        gp = np.stack([_to_unit_np(gallery_map[l]["palm_embedding"]) for l in gal_labels])
        gd = np.stack([_to_unit_np(gallery_map[l]["dorsal_embedding"]) for l in gal_labels])

    for start in range(0, len(probes), chunk):
        block = probes[start:start + chunk]
        pp = np.stack([_to_unit_np(p["palm_embedding"]) for _, p in block])
        pdor = np.stack([_to_unit_np(p["dorsal_embedding"]) for _, p in block])
        SP = pp @ gp.T
        SD = pdor @ gd.T
        if norm_stats is not None:
            SPf = _normalize(SP, norm_stats["palm"])
            SDf = _normalize(SD, norm_stats["dorsal"])
        else:
            SPf, SDf = SP, SD
        SF = alpha * SPf + (1.0 - alpha) * SDf

        for r, (true_label, probe) in enumerate(block):
            probe_side = true_label.rsplit("_", 1)[-1]
            cand = np.nonzero(gal_sides == probe_side)[0]
            f = SF[r, cand]
            true_pos = int(np.nonzero(cand == label_index[true_label])[0][0])
            tf = f[true_pos]
            rank = 1 + int((f > tf).sum()) + int((f[:true_pos] == tf).sum())

            if len(f) > 1:
                top2 = np.partition(f, -2)[-2:]
                margin_top1_top2 = float(top2[1] - top2[0])
            else:
                margin_top1_top2 = float("nan")
            if len(f) > 1 and abs(margin_top1_top2) < debug_tie_threshold:
                n_tied_top1 += 1

            rank1 += int(rank <= 1)
            rank5 += int(rank <= 5)
            rank10 += int(rank <= 10)
            reciprocal_sum += 1.0 / rank

            details.append({
                "system": name, "probe_subject": probe["subject"], "probe_side": probe["side"],
                "gallery_side": probe["side"], "rank": rank,
                "predicted_subject": gal_labels[int(cand[int(np.argmax(f))])],
                "top1_correct": bool(rank == 1), "top5_correct": bool(rank <= 5),
                "top10_correct": bool(rank <= 10),
                "top1_palm": gal_labels[int(cand[int(np.argmax(SP[r, cand]))])],
                "top1_dorsal": gal_labels[int(cand[int(np.argmax(SD[r, cand]))])],
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
            "degli embedding."
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
    rng = random.Random(seed)
    subs = sorted(subjects)
    rng.shuffle(subs)

    n_known = int(round(known_fraction * len(subs)))
    n_known = min(max(n_known, 2), len(subs) - 2)
    known = subs[:n_known]
    unknown = subs[n_known:]

    n_cal = int(round(cal_fraction * len(unknown)))
    n_cal = min(max(n_cal, 1), len(unknown) - 1)
    return set(known), set(unknown[:n_cal]), set(unknown[n_cal:])


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
    labels = {"all": "probe genuini reali+augmentati", "real_only": "probe genuini SOLO reali"}
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
    all_pairs = list(genuine) + list(impostor)
    sc = pair_scores(embeddings, all_pairs, alpha, norm_stats=norm_stats)
    n_g = len(genuine)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "subject1", "side1", "subject2", "side2",
                         "palm_similarity", "dorsal_similarity", "fused_similarity"])
        for k, (i, j) in enumerate(all_pairs):
            writer.writerow([
                "genuine" if k < n_g else "impostor",
                embeddings[i]["subject"], embeddings[i]["side"],
                embeddings[j]["subject"], embeddings[j]["side"],
                float(sc["palm"][k]), float(sc["dorsal"][k]), float(sc["fused"][k]),
            ])


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

    if args.preprocessed_only:
        # Modalita' "solo preprocessed": nessun accesso al dataset raw,
        # nessuna augmentation, nessun preprocessing. I campioni vengono
        # costruiti direttamente dai file gia' presenti in --preprocessed_dir.
        (processed_pairs, records, missing_pairs,
         preprocess_failures, unparsed_files) = discover_preprocessed_samples(preprocessed_dir)
        raw_pairs = processed_pairs
        aug_report = {"hands_augmented": [], "n_new_raw_images": 0}
        n_augmented = sum(1 for r in records if r.get("is_augmented"))

        print("\n" + "=" * 72)
        print("NUOVO DATASET - EVALUATION (v4) - SOLO PREPROCESSED, NO AUGMENTATION")
        print("=" * 72)
        print(f"Cartella preprocessed:        {preprocessed_dir}")
        print(f"File preprocessati trovati:   {len(records)}")
        print(f"Coppie multimodali complete:  {len(processed_pairs)}")
        print(f"Coppie incomplete:            {len(missing_pairs)}")
        print(f"Coppie scartate (file mancanti): {len(preprocess_failures)}")
        print(f"Soggetti:                     {len(set(x['subject'] for x in processed_pairs))}")
        if unparsed_files:
            print(f"[!] {len(unparsed_files)} file ignorati perche' il nome non contiene "
                  f"soggetto PXXX e lato L/R. Esempio: {unparsed_files[0]}")
    else:
        records = discover_raw_samples(args.data_dir)
        raw_pairs, missing_pairs = build_raw_pairs(records)

        print("\n" + "=" * 72)
        print("NUOVO DATASET - EVALUATION (v4)")
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
        if args.no_augmentation:
            aug_report = {"hands_augmented": [], "n_new_raw_images": 0}
            print("\n[i] Data augmentation DISABILITATA (--no_augmentation).")
        else:
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

    if not args.preprocessed_only:
        processed_pairs, preprocess_failures = build_processed_pairs(preprocessed_dir, raw_pairs)
    print(f"\nCampioni multimodali utilizzabili: {len(processed_pairs)}")

    if args.no_augmentation or args.preprocessed_only:
        per_hand = {}
        for p in processed_pairs:
            per_hand[(p["subject"], p["side"])] = per_hand.get((p["subject"], p["side"]), 0) + 1
        n_single = sum(1 for c in per_hand.values() if c < 2)
        if n_single:
            print(f"[!] {n_single} mani su {len(per_hand)} hanno un solo scatto: "
                  "non generano coppie genuine (nessuna augmentation attiva).")

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
            norm_stats=norm_stats,
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
            "dataset_type": "external_new_dataset",
            "input_format": ("preprocessed: *_palm_hand.png / *_dorsal_hand.png (PXXX, L/R, scatto)"
                             if args.preprocessed_only
                             else "PXXX_[SESS]_L/R_palmar/dorsal[_augN]_processed.png"),
            "preprocessed_only": bool(args.preprocessed_only),
            "augmentation_disabled": bool(args.no_augmentation or args.preprocessed_only),
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
            "open_set_enabled": open_set_results is not None,
            "open_set_repeats": args.openset_repeats if open_set_results is not None else None,
            "open_set_known_fraction": args.openset_known_fraction if open_set_results is not None else None,
            "open_set_unknown_samples": args.openset_unknown_samples if open_set_results is not None else None,
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
        "open_set": _json_safe(open_set_results) if open_set_results is not None else None,
    }

    save_json(results_dir / "summary.json", summary)

    with open(results_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("RISULTATI NUOVO DATASET - PALMO + DORSO (v4)\n")
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

        if open_set_results is not None:
            f.write("\nIDENTIFICATION OPEN-SET\n" + "-" * 72 + "\n")
            for line in format_open_set_summary(open_set_results):
                f.write(line + "\n")

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
        description="Test multimodale palmo+dorso - v4, niente fallback L/R, con open-set."
    )

    parser.add_argument("--data_dir", default=None,
                         help="Cartella con il dataset RAW di input (sola lettura). "
                              "Non serve con --preprocessed_only.")
    parser.add_argument(
        "--preprocessed_only", action="store_true",
        help="Usa DIRETTAMENTE i file gia' preprocessati in --preprocessed_dir: "
             "non legge il raw, non fa preprocessing e non fa data augmentation. "
             "Richiede --preprocessed_dir; --data_dir non e' necessario.",
    )
    parser.add_argument(
        "--no_augmentation", action="store_true",
        help="Disabilita la data augmentation automatica (sempre implicita con "
             "--preprocessed_only).",
    )
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
    parser.add_argument("--skip_openset", action="store_true",
                         help="Salta la valutazione di identificazione open-set.")
    parser.add_argument("--openset_repeats", type=int, default=10,
                         help="Numero di split casuali subject-disjoint known/unknown (default: 10).")
    parser.add_argument("--openset_known_fraction", type=float, default=0.5,
                         help="Frazione di soggetti iscritti in gallery (known); "
                              "i restanti sono unknown (default: 0.5).")
    parser.add_argument("--openset_cal_fraction", type=float, default=0.5,
                         help="Frazione degli unknown usata per calibrare la soglia; "
                              "il resto e' usato solo per misurare FPIR (default: 0.5).")
    parser.add_argument("--openset_target_fpirs", default="0.01,0.05,0.10",
                         help="FPIR-obiettivo (separati da virgola) a cui calibrare la soglia.")
    parser.add_argument("--openset_unknown_samples", choices=["real", "all"], default="real",
                         help="Scatti degli unknown usati come probe: solo reali (default) "
                              "o anche augmentati (quasi-duplicati, gonfiano n).")
    parser.add_argument("--max_impostor_pairs", type=int, default=None,
                         help="Numero di coppie impostor CAMPIONATE per la verifica 1:1 "
                              "(default: 500000). Mai enumerate tutte le N^2/2 coppie.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_preprocessing", action="store_true")
    parser.add_argument("--clean_preprocessed", action="store_true")

    args = parser.parse_args()

    if args.preprocessed_only:
        if args.preprocessed_dir is None:
            parser.error("--preprocessed_only richiede --preprocessed_dir.")
        args.skip_preprocessing = True
        args.no_augmentation = True
    elif args.data_dir is None:
        parser.error("--data_dir e' obbligatorio (a meno di usare --preprocessed_only).")

    if args.min_samples_per_hand < 2:
        raise ValueError(
            "--min_samples_per_hand deve essere >= 2: senza almeno 2 scatti "
            "per mano non esiste alcuna coppia genuine valida (e non c'e' "
            "alcun fallback L-vs-R a cui ricorrere)."
        )

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha deve essere compreso tra 0 e 1.")

    if args.openset_repeats < 1:
        raise ValueError("--openset_repeats deve essere >= 1.")

    evaluate_dataset(args)


if __name__ == "__main__":
    main()