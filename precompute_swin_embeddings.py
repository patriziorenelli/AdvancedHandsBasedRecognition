"""
============================================================
PRECOMPUTE_SWIN_EMBEDDINGS
============================================================
Calcola UNA VOLTA SOLA l'embedding del backbone Swin-Tiny (sempre
congelato in questa pipeline: freeze_swin=True) per ogni immagine
"*_dorsal_hand.png" presente in --data_dir, e la salva in un unico
file .npz.

Perche' qui il guadagno e' ANCORA MAGGIORE che nella pipeline palm
------------------------------------------------------------------
Il branch texture dorsale applica, PRIMA del backbone Swin, la
funzione enhance_veins() (CLAHE aggressiva + filtro di Frangi
multiscala per esaltare le vene): e' un preprocessing handcrafted
via skimage, tipicamente lento su CPU quanto (o piu' di) un forward
del backbone stesso. Con freeze_swin=True, sia enhance_veins() sia
il forward Swin producono SEMPRE lo stesso risultato per la stessa
immagine, ma vengono ricalcolati da zero ad ogni epoca, in ogni
inner fold, in ogni candidato, in ogni outer fold.

Questo script fa il preprocessing (enhance_veins + normalizzazione,
SENZA le augmentation random del training) e il forward Swin una
sola volta per immagine, e salva il vettore risultante.

Uso
---
python precompute_swin_embeddings.py --data_dir "D:/Users/Patrizio/Desktop/Tesi/dataset_preprocessed" --out "D:/Users/Patrizio/Desktop/Tesi/swin_cache.npz"

Lo script stampa una ETA live basata sulla velocita' reale osservata.
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

from dorsal_core import cfg, dorsal_hand_transform, DorsalTextureBranch


def find_dorsal_hand_images(data_dir: Path):
    """Stessa logica di scoperta file di DorsalBiometricDataset, ma
    restituisce solo i path delle immagini dorsal_hand."""
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
            if not knuckle_paths:
                continue
            paths.append(hand_path.resolve())
    return paths


def load_batch(paths, tf):
    imgs = []
    for p in paths:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        imgs.append(tf(img))  # tf include enhance_veins() (vedi dorsal_hand_transform)
    return torch.stack(imgs, dim=0)


def main():
    ap = argparse.ArgumentParser(description="Precalcola le embedding Swin-Tiny (congelato) + enhance_veins")
    ap.add_argument("--data_dir", required=True, help="stesso --data_dir usato per nested_cv")
    ap.add_argument("--out", default="swin_cache.npz", help="file .npz di output")
    ap.add_argument("--batch_size", type=int, default=16,
                     help="batch size per il precompute (puo' essere diverso da quello del training)")
    ap.add_argument("--num_threads", type=int, default=0,
                     help="torch.set_num_threads; 0 = default di PyTorch")
    args = ap.parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    data_dir = Path(args.data_dir)
    print(f"Scansione dataset in: {data_dir}")
    hand_paths = find_dorsal_hand_images(data_dir)
    n = len(hand_paths)
    print(f"Trovate {n} immagini dorsal_hand da processare "
          f"(enhance_veins + Swin-Tiny per ciascuna).")
    if n == 0:
        raise SystemExit("Nessuna immagine trovata: controlla --data_dir")

    # Transform SENZA augmentation (train=False), MA con enhance_veins()
    # incluso (fa parte di dorsal_hand_transform): l'embedding deve
    # essere deterministica, e' quella riusata identica in tutti i fold.
    tf = dorsal_hand_transform(train=False)

    print("Carico il backbone Swin-Tiny (pretrained, congelato)...")
    device = torch.device("cpu")  # niente CUDA disponibile su questa macchina
    branch = DorsalTextureBranch(out_dim=cfg.TEXTURE_EMBED_DIM, freeze_backbone=True).to(device)
    branch.eval()
    feat_dim = branch.backbone.num_features
    print(f"Dimensione embedding backbone: {feat_dim}")

    all_embeds = np.zeros((n, feat_dim), dtype=np.float32)
    all_paths = np.empty((n,), dtype=object)

    batch_size = args.batch_size
    n_batches = (n + batch_size - 1) // batch_size

    t_start = time.time()
    processed = 0
    with torch.no_grad():
        for bi in range(n_batches):
            lo, hi = bi * batch_size, min(n, (bi + 1) * batch_size)
            batch_paths = hand_paths[lo:hi]

            t0 = time.time()
            x = load_batch(batch_paths, tf).to(device)  # qui gira anche enhance_veins (Frangi)
            feats = branch.backbone(x)  # (B, feat_dim), backbone congelato
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
    np.savez_compressed(out_path, paths=all_paths, embeds=all_embeds)
    total_min = (time.time() - t_start) / 60
    print(f"\nFatto. Salvate {n} embedding in '{out_path}' ({total_min:.1f} min totali).")
    print(f"Dimensione file: {out_path.stat().st_size / 1e6:.1f} MB")
    print(
        "\nLancia ora il nested CV aggiungendo:\n"
        f"  --swin_cache_path {out_path.resolve()}"
    )


if __name__ == "__main__":
    main()