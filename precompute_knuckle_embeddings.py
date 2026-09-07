"""
============================================================
PRECOMPUTE_KNUCKLE_EMBEDDINGS
============================================================
Calcola UNA VOLTA SOLA le feature del backbone MobileNetV3-Large
(prima del layer 'proj', che invece resta sempre allenabile) per
ogni crop "*_dorsal_<nocca>.png" presente in --data_dir.

ATTENZIONE - quando ha senso usare questa cache
-------------------------------------------------
Il layer 'proj' sopra il backbone MobileNet NON viene mai
congelato in questo codice, quindi la cache e' valida SOLO se il
training che la userà ha SEMPRE freeze_mobilenet=True (il
backbone stesso non si allena). Se anche un solo candidato della
grid usa freeze_mobilenet=False, la cache per quel candidato
sarebbe sbagliata: dorsal_run.py lo verifica e si rifiuta di
partire in quel caso.

Uso
---
python precompute_knuckle_embeddings.py --data_dir "D:/Users/Patrizio/Desktop/Tesi/dataset_preprocessed" --out "D:/Users/Patrizio/Desktop/Tesi/knuckle_cache.npz"
============================================================
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dorsal_core import cfg, knuckle_transform, KnuckleMobileNetBranch


def find_knuckle_crops(data_dir: Path):
    paths = []
    subject_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    for sdir in subject_dirs:
        for meta_path in sorted(sdir.glob("*_metadata.json")):
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            if not meta.get("is_dorsal", False):
                continue
            base = meta_path.name.replace("_metadata.json", "")
            hand_path = sdir / f"{base}_dorsal_hand.png"
            if not hand_path.exists():
                continue
            knuckle_paths = sorted(sdir.glob(f"{base}_dorsal_*.png"))
            knuckle_paths = [p for p in knuckle_paths if p.name != hand_path.name]
            paths.extend(p.resolve() for p in knuckle_paths)
    return paths


def load_batch(paths, tf):
    imgs = []
    for p in paths:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        imgs.append(tf(img))
    return torch.stack(imgs, dim=0)


def main():
    ap = argparse.ArgumentParser(description="Precalcola le feature MobileNetV3 (congelato) delle nocche dorsali")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="knuckle_cache.npz")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_threads", type=int, default=0)
    args = ap.parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    data_dir = Path(args.data_dir)
    print(f"Scansione dataset in: {data_dir}")
    knuckle_paths = find_knuckle_crops(data_dir)
    n = len(knuckle_paths)
    print(f"Trovati {n} crop di nocche da processare.")
    if n == 0:
        raise SystemExit("Nessun crop trovato: controlla --data_dir")

    tf = knuckle_transform(train=False)  # senza augmentation, deterministico

    print("Carico il backbone MobileNetV3-Large (pretrained)...")
    device = torch.device("cpu")
    branch = KnuckleMobileNetBranch(n_knuckles=12, freeze_backbone=True).to(device)
    branch.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        dummy_feat = branch.cnn.avgpool(branch.cnn.features(dummy)).flatten(1)
        feat_dim = dummy_feat.shape[1]
    print(f"Dimensione feature backbone (pre-proj): {feat_dim}")

    all_embeds = np.zeros((n, feat_dim), dtype=np.float32)
    all_paths = np.empty((n,), dtype=object)

    batch_size = args.batch_size
    n_batches = (n + batch_size - 1) // batch_size

    t_start = time.time()
    processed = 0
    with torch.no_grad():
        for bi in range(n_batches):
            lo, hi = bi * batch_size, min(n, (bi + 1) * batch_size)
            batch_paths = knuckle_paths[lo:hi]

            t0 = time.time()
            x = load_batch(batch_paths, tf).to(device)
            feats = branch.cnn.avgpool(branch.cnn.features(x)).flatten(1)
            feats = feats.cpu().numpy().astype(np.float32)
            dt = time.time() - t0

            all_embeds[lo:hi] = feats
            for i, p in enumerate(batch_paths):
                all_paths[lo + i] = str(p)

            processed = hi
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta_s = (n - processed) / rate if rate > 0 else float("nan")
            print(
                f"[{processed}/{n}] batch {bi+1}/{n_batches} "
                f"({dt:.2f}s/batch, {dt/len(batch_paths):.3f}s/img) "
                f"- trascorso {elapsed/60:.1f} min - ETA {eta_s/60:.1f} min",
                flush=True,
            )

    out_path = Path(args.out)
    np.savez_compressed(out_path, paths=np.array(all_paths, dtype=str), embeds=all_embeds)
    total_min = (time.time() - t_start) / 60
    print(f"\nFatto. Salvate {n} embedding in '{out_path}' ({total_min:.1f} min totali).")
    print(f"Dimensione file: {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"\nLancia ora il nested CV aggiungendo:\n  --knuckle_cache_path {out_path.resolve()}")


if __name__ == "__main__":
    main()