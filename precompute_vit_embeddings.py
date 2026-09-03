"""
============================================================
PRECOMPUTE_VIT_EMBEDDINGS
============================================================
Calcola UNA VOLTA SOLA l'embedding del backbone ViT-S/14 DINOv2
(sempre congelato in questa pipeline: freeze_vit=True) per ogni
immagine "*_palm_hand.png" presente in --data_dir, e la salva in
un unico file .npz.

Perche' serve
-------------
Nel nested CV il branch ViT non viene mai allenato (freeze_vit=True
in tutta la grid search), quindi la sua uscita per una data immagine
e' SEMPRE la stessa, in ogni epoca, in ogni inner fold, in ogni
candidato, in ogni outer fold. Nonostante questo, il forward del ViT
viene ricalcolato da zero decine di migliaia di volte durante il
nested CV: e' il calcolo piu' ridondante (e su CPU anche il piu'
lento) di tutta la pipeline.

Calcolando l'embedding una sola volta per immagine e salvandola su
disco, il training successivo (palm_run.py nested_cv --vit_cache_path ...)
salta del tutto il forward del ViT e usa direttamente il vettore
gia' pronto: quello che resta da allenare per il branch texture e'
solo il piccolo MLP (Linear-GELU-Dropout-Linear) sopra l'embedding.

NB: la cache viene calcolata SENZA le augmentation (ColorJitter/Blur)
che nel training vengono normalmente applicate a palm_hand. Questo
e' un compromesso consapevole per la velocita': il backbone e'
pre-addestrato (DINOv2) e congelato, quindi il beneficio dell'augmentation
sulla sua uscita e' comunque marginale rispetto al costo.

Uso
---
python precompute_vit_embeddings.py --data_dir "D:/Users/Patrizio/Desktop/Tesi/dataset_preprocessed" --out vit_cache.npz

Lo script stampa periodicamente una ETA (tempo stimato rimanente)
calcolata sul tempo reale osservato sulla TUA macchina, non su una
stima teorica: e' quindi affidabile dopo i primi batch.
============================================================
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from palm_core import cfg, rgb_transform, TextureBranch


def find_palm_hand_images(data_dir: Path):
    """Stessa logica di scoperta file di PalmBiometricDataset, ma
    restituisce solo i path delle immagini palm_hand (con controllo
    is_dorsal via metadata, per restare coerenti col dataset reale)."""
    import json

    paths = []
    subject_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    for sdir in subject_dirs:
        for meta_path in sorted(sdir.glob("*_metadata.json")):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except UnicodeDecodeError:
                try:
                    with open(meta_path, "r", encoding="utf-16") as f:
                        meta = json.load(f)
                except UnicodeDecodeError:
                    with open(meta_path, "r", encoding="utf-8-sig") as f:
                        meta = json.load(f)
            if meta.get("is_dorsal", False):
                continue
            base = meta_path.name.replace("_metadata.json", "")
            hand_path = sdir / f"{base}_palm_hand.png"
            roi_path = sdir / f"{base}_palm_roi.png"
            if not (hand_path.exists() and roi_path.exists()):
                continue
            knuckle_paths = [p for p in sorted(sdir.glob(f"{base}_palm_*.png"))
                              if p.name not in (hand_path.name, roi_path.name)]
            if not knuckle_paths:
                continue
            paths.append(hand_path.resolve())
    return paths


def load_batch(paths, tf):
    imgs = []
    for p in paths:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        imgs.append(tf(img))
    return torch.stack(imgs, dim=0)


def main():
    ap = argparse.ArgumentParser(description="Precalcola le embedding ViT-S/14 DINOv2 (congelato)")
    ap.add_argument("--data_dir", required=True, help="stesso --data_dir usato per nested_cv")
    ap.add_argument("--out", default="vit_cache.npz", help="file .npz di output")
    ap.add_argument("--batch_size", type=int, default=16,
                     help="batch size per il precompute (puo' essere diverso da quello del training)")
    ap.add_argument("--num_threads", type=int, default=0,
                     help="torch.set_num_threads; 0 = lascia il default di PyTorch")
    args = ap.parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    data_dir = Path(args.data_dir)
    print(f"Scansione dataset in: {data_dir}")
    hand_paths = find_palm_hand_images(data_dir)
    n = len(hand_paths)
    print(f"Trovate {n} immagini palm_hand da processare.")
    if n == 0:
        raise SystemExit("Nessuna immagine trovata: controlla --data_dir")

    # Transform SENZA augmentation (train=False): l'embedding deve essere
    # deterministica, e' quella che verra' riusata identica in tutti i fold.
    tf = rgb_transform(train=False)

    print("Carico il backbone ViT-S/14 DINOv2 (pretrained, congelato)...")
    device = torch.device("cpu")  # niente CUDA disponibile su questa macchina
    branch = TextureBranch(out_dim=cfg.TEXTURE_EMBED_DIM, freeze_backbone=True).to(device)
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
            x = load_batch(batch_paths, tf).to(device)
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
        f"  --vit_cache_path {out_path.resolve()}"
    )


if __name__ == "__main__":
    main()