"""
============================================================
SUBJECT_SPLIT - Split per soggetti: TRAIN / VAL / HELD-OUT
============================================================
Modulo leggero (solo numpy + stdlib, nessuna dipendenza da torch) pensato per
essere condiviso tra la pipeline PALMO e la pipeline DORSO, cosi' entrambi i
modelli vengono addestrati ESATTAMENTE sugli stessi soggetti e lo stesso gruppo
di soggetti HELD-OUT resta "mai visto" da entrambi (requisito fondamentale per
valutare correttamente la fusione multimodale).

Gruppi (tutti disgiunti, split sempre PER SOGGETTO):

  train_subjects    -> addestramento dei modelli
  val_subjects      -> validation open-set: SOLO scelta dell'epoca migliore / early stopping
  heldout_subjects  -> MAI usati in training, validation, nested CV, scelta di
                       iperparametri o epoche. Riservati alla valutazione finale
                       con la pipeline multimodale (verifica 1:1, identificazione
                       1:N, identificazione open-set).

Dentro gli held-out viene gia' proposta una divisione per l'open-set:

  heldout_enrolled_subjects -> soggetti "noti" (gallery / enrollment)
  heldout_unknown_subjects  -> soggetti "sconosciuti" (mai in gallery: impostori open-set)

Lo split viene creato UNA volta, salvato in un JSON (default
./splits/subject_split_new.json) e poi sempre ricaricato: il file e' la fonte
di verita'. Se esiste gia' non viene mai rigenerato, a meno di --regenerate_split
(il vecchio file viene in ogni caso salvato come backup).

Proprieta' utili:
  - e' deterministico: stessi soggetti + stesso seed + stesso heldout_ratio
    => stessi held-out, anche cambiando val_ratio o known_ratio;
  - l'insieme dei soggetti nel JSON viene confrontato con quello presente nella
    cartella dati: se non coincide viene sollevato un errore (niente split
    "silenziosamente" incoerenti tra palmo e dorso).

Uso da riga di comando (opzionale, lo split viene creato in automatico al primo
training):
    python subject_split.py --data_dir dataset_preprocessed
============================================================
"""

from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SPLIT_VERSION = 1

DEFAULT_HELDOUT_RATIO = 0.25   # quota di soggetti riservata alla valutazione finale
DEFAULT_VAL_RATIO = 0.12       # quota dei soggetti NON held-out usata come validation
DEFAULT_KNOWN_RATIO = 0.50     # quota degli held-out "enrolled" (il resto = sconosciuti open-set)
DEFAULT_SEED = 42
DEFAULT_SPLIT_FILE = Path("./splits/subject_split_new.json")

_PARTITION_KEYS = ("train_subjects", "val_subjects", "heldout_subjects")


# ============================================================
# UTILITA'
# ============================================================
def list_subjects(data_dir):
    """Ritorna gli ID soggetto ordinati presenti nel dataset (una cartella = un soggetto)."""
    return sorted(d.name for d in Path(data_dir).iterdir() if d.is_dir())


def subjects_fingerprint(subject_ids) -> str:
    """Impronta SHA-1 dell'insieme dei soggetti (indipendente dall'ordine)."""
    return hashlib.sha1("\n".join(sorted(subject_ids)).encode("utf-8")).hexdigest()


# ============================================================
# CREAZIONE DELLO SPLIT
# ============================================================
def make_holdout_split(subject_ids, heldout_ratio: float = DEFAULT_HELDOUT_RATIO,
                       val_ratio: float = DEFAULT_VAL_RATIO,
                       known_ratio: float = DEFAULT_KNOWN_RATIO,
                       seed: int = DEFAULT_SEED) -> dict:
    """
    Crea lo split per soggetti. I soggetti sono ordinati, mescolati con
    RandomState(seed) e tagliati in sequenza:

        [ held-out | validation | train ]

    Per questo gli held-out dipendono SOLO da (soggetti, seed, heldout_ratio).
    """
    subjects = sorted(set(subject_ids))
    n = len(subjects)

    if not 0.0 < heldout_ratio < 1.0:
        raise ValueError(f"heldout_ratio deve essere in (0,1), ricevuto {heldout_ratio}")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio deve essere in (0,1), ricevuto {val_ratio}")
    if not 0.0 < known_ratio < 1.0:
        raise ValueError(f"openset_known_ratio deve essere in (0,1), ricevuto {known_ratio}")

    n_heldout = int(round(n * heldout_ratio))
    n_dev = n - n_heldout
    n_val = int(round(n_dev * val_ratio))
    n_train = n_dev - n_val
    n_enrolled = int(round(n_heldout * known_ratio))
    n_unknown = n_heldout - n_enrolled

    if n_enrolled < 2 or n_unknown < 2:
        raise ValueError(
            f"Held-out troppo piccolo ({n_heldout} soggetti su {n}): servono almeno 2 soggetti "
            f"enrolled e 2 sconosciuti per l'open-set (ottenuti {n_enrolled}/{n_unknown}). "
            f"Aumenta heldout_ratio."
        )
    if n_val < 2:
        raise ValueError(f"Validation troppo piccola ({n_val} soggetti): aumenta val_ratio.")
    if n_train < 2:
        raise ValueError(f"Training troppo piccolo ({n_train} soggetti): riduci heldout_ratio/val_ratio.")

    rng = np.random.RandomState(seed)
    shuffled = [subjects[i] for i in rng.permutation(n)]

    heldout = shuffled[:n_heldout]
    val = shuffled[n_heldout:n_heldout + n_val]
    train = shuffled[n_heldout + n_val:]
    enrolled = heldout[:n_enrolled]
    unknown = heldout[n_enrolled:]

    split = {
        "version": SPLIT_VERSION,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
        "heldout_ratio": float(heldout_ratio),
        "val_ratio": float(val_ratio),
        "known_ratio": float(known_ratio),
        "n_subjects_total": n,
        "subjects_fingerprint": subjects_fingerprint(subjects),
        "n_train": len(train),
        "n_val": len(val),
        "n_heldout": len(heldout),
        "n_heldout_enrolled": len(enrolled),
        "n_heldout_unknown": len(unknown),
        "train_subjects": sorted(train),
        "val_subjects": sorted(val),
        "heldout_subjects": sorted(heldout),
        "heldout_enrolled_subjects": sorted(enrolled),
        "heldout_unknown_subjects": sorted(unknown),
    }
    assert_valid_split(split)
    return split


def assert_valid_split(split: dict) -> None:
    """Controlla che i gruppi siano disgiunti e coerenti tra loro."""
    train, val, held = (set(split[k]) for k in _PARTITION_KEYS)
    enrolled = set(split["heldout_enrolled_subjects"])
    unknown = set(split["heldout_unknown_subjects"])

    for (name_a, a), (name_b, b) in [
        (("train", train), ("val", val)),
        (("train", train), ("heldout", held)),
        (("val", val), ("heldout", held)),
        (("heldout_enrolled", enrolled), ("heldout_unknown", unknown)),
    ]:
        common = a & b
        if common:
            raise ValueError(
                f"Split non valido: {name_a} e {name_b} condividono "
                f"{len(common)} soggetti (es. {sorted(common)[:3]})"
            )
    if enrolled | unknown != held:
        raise ValueError("Split non valido: enrolled + unknown devono coincidere con gli held-out")


def dev_subjects(split: dict) -> list:
    """Soggetti NON held-out (train + val): l'unico insieme su cui si puo' fare nested CV."""
    return sorted(set(split["train_subjects"]) | set(split["val_subjects"]))


def assert_disjoint_from_heldout(subjects, split: dict, name: str = "subjects") -> None:
    """Rete di sicurezza anti-leakage: nessun soggetto held-out puo' finire in training/validation/CV."""
    leaked = set(subjects) & set(split["heldout_subjects"])
    if leaked:
        raise RuntimeError(
            f"LEAKAGE: {len(leaked)} soggetti held-out presenti in '{name}' "
            f"(es. {sorted(leaked)[:5]}). Interrompo."
        )


# ============================================================
# SALVATAGGIO / CARICAMENTO
# ============================================================
def save_split(split: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)   # scrittura atomica
    return path


def load_split(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        split = json.load(f)
    missing = [k for k in (*_PARTITION_KEYS, "heldout_enrolled_subjects", "heldout_unknown_subjects")
               if k not in split]
    if missing:
        raise ValueError(f"File di split {path} incompleto, chiavi mancanti: {missing}")
    assert_valid_split(split)
    return split


def get_or_create_split(data_dir, split_path=DEFAULT_SPLIT_FILE,
                        heldout_ratio: float = DEFAULT_HELDOUT_RATIO,
                        val_ratio: float = DEFAULT_VAL_RATIO,
                        known_ratio: float = DEFAULT_KNOWN_RATIO,
                        seed: int = DEFAULT_SEED,
                        regenerate: bool = False) -> dict:
    """
    Se split_path esiste lo carica (e verifica che coincida con i soggetti presenti
    in data_dir), altrimenti lo crea e lo salva. Con regenerate=True ne crea uno
    nuovo salvando prima quello vecchio come backup.
    """
    split_path = Path(split_path)
    universe = list_subjects(data_dir)
    if not universe:
        raise RuntimeError(f"Nessun soggetto (sottocartella) trovato in {data_dir}")

    if split_path.exists() and not regenerate:
        split = load_split(split_path)
        stored = set(split["train_subjects"]) | set(split["val_subjects"]) | set(split["heldout_subjects"])
        current = set(universe)
        if stored != current:
            only_file = sorted(stored - current)[:5]
            only_dir = sorted(current - stored)[:5]
            raise ValueError(
                f"Lo split in {split_path} non coincide con i soggetti di {data_dir}: "
                f"{len(stored - current)} solo nel file (es. {only_file}), "
                f"{len(current - stored)} solo nella cartella (es. {only_dir}). "
                f"Usa un altro --split_file oppure --regenerate_split (ATTENZIONE: i modelli gia' "
                f"addestrati con il vecchio split non sarebbero piu' coerenti)."
            )
        diffs = []
        for key, requested in (("heldout_ratio", heldout_ratio), ("val_ratio", val_ratio),
                               ("known_ratio", known_ratio), ("seed", seed)):
            if key in split and abs(float(split[key]) - float(requested)) > 1e-9:
                diffs.append(f"{key}: file={split[key]} richiesto={requested}")
        if diffs:
            print(f"[split] ATTENZIONE: parametri richiesti diversi da quelli del file esistente "
                  f"({'; '.join(diffs)}). Vale il file {split_path} (usa --regenerate_split per ricrearlo).")
        print(f"[split] Caricato split esistente: {split_path}")
        return split

    if split_path.exists() and regenerate:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = split_path.with_name(f"{split_path.stem}.bak_{ts}{split_path.suffix}")
        split_path.rename(backup)
        print(f"[split] Split precedente salvato come backup: {backup}")

    split = make_holdout_split(universe, heldout_ratio, val_ratio, known_ratio, seed)
    split["data_dir"] = str(data_dir)
    save_split(split, split_path)
    print(f"[split] Creato nuovo split: {split_path}")
    return split


def split_summary(split: dict) -> str:
    return (
        f"Split soggetti (seed={split.get('seed')}, heldout_ratio={split.get('heldout_ratio')}): "
        f"totale={split.get('n_subjects_total', '?')} | "
        f"train={len(split['train_subjects'])} | val={len(split['val_subjects'])} | "
        f"HELD-OUT={len(split['heldout_subjects'])} "
        f"(enrolled={len(split['heldout_enrolled_subjects'])}, "
        f"unknown open-set={len(split['heldout_unknown_subjects'])})"
    )


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Crea (o mostra) lo split per soggetti train/val/held-out condiviso tra palmo e dorso"
    )
    ap.add_argument("--data_dir", required=True, help="output di preProcessing.py (una cartella per soggetto)")
    ap.add_argument("--split_file", default=str(DEFAULT_SPLIT_FILE))
    ap.add_argument("--heldout_ratio", type=float, default=DEFAULT_HELDOUT_RATIO)
    ap.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--openset_known_ratio", type=float, default=DEFAULT_KNOWN_RATIO)
    ap.add_argument("--split_seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--regenerate_split", action="store_true",
                    help="ricrea lo split (il file esistente viene salvato come backup)")
    args = ap.parse_args()

    split = get_or_create_split(
        args.data_dir, args.split_file, heldout_ratio=args.heldout_ratio,
        val_ratio=args.val_ratio, known_ratio=args.openset_known_ratio,
        seed=args.split_seed, regenerate=args.regenerate_split,
    )
    print(split_summary(split))


if __name__ == "__main__":
    main()