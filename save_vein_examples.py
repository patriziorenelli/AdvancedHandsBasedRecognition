"""
save_vein_pipeline_steps.py - Confronto originale vs step della pipeline dorsale (fedele alla pipeline).

Serve solo il path del dataset originale (11k Hands). Per ogni immagine dorsale campionata:
  1. esegue il VERO preProcessing.py (hand detection, normalizzazione geometrica,
     illuminazione + CLAHE 1.5, crop 224) -> <out_dir>/preprocessed/...
  2. applica sull'output (esattamente come DorsalBiometricDataset) gli stadi di enhance_veins:
        CLAHE forte (L, Lab) -> canale verde -> Frangi (vesselness) -> blend finale
  3. salva il pannello  originale | preprocessing | CLAHE forte | verde | Frangi | enhance_veins
     e l'immagine finale (quella che entra nello Swin prima della normalizzazione ImageNet).

Gli stadi sono ricalcolati con gli stessi parametri di default di dorsal_core.enhance_veins
e verificati uguali al risultato reale (altrimenti stampa [ATTENZIONE]).

Esecuzione (dalla cartella con dorsal_core.py e preProcessing.py):
  python save_vein_pipeline_steps.py --dataset_dir "D:\\Users\\Patrizio\\Desktop\\Tesi\\Hands_11k"

Se immagini e CSV non sono in <dataset_dir>/Hands e <dataset_dir>/HandInfo.csv usa --images_dir / --csv.
"""
import argparse
import inspect
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi

import preProcessing as PP                      # preprocessing REALE
from dorsal_core import enhance_veins           # enhancement REALE

PANEL_H = 224


def to_rgb3(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img


def fit_height(img, h=PANEL_H):
    s = h / img.shape[0]
    return cv2.resize(img, (max(1, int(round(img.shape[1] * s))), h), interpolation=cv2.INTER_AREA)


def label(img_rgb, text):
    out = img_rgb.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def vein_stages(img_rgb):
    """Replica passo-passo enhance_veins (stessi parametri di default). Ritorna dict di stadi RGB uint8."""
    p = inspect.signature(enhance_veins).parameters
    clip, scales, weight = p["clahe_clip"].default, p["vessel_scales"].default, p["vessel_weight"].default

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    L_eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(L)
    contrast_rgb = cv2.cvtColor(cv2.merge([L_eq, a, b]), cv2.COLOR_LAB2RGB)

    green = contrast_rgb[:, :, 1].astype(np.float64) / 255.0
    vess = frangi(1.0 - green, sigmas=scales, black_ridges=False)
    if vess.max() > 1e-8:
        vess = vess / vess.max()
    vess_rgb = cv2.cvtColor((vess * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    final = np.clip(cv2.addWeighted(contrast_rgb, 1.0 - weight * 0.5, vess_rgb, weight * 0.5, 0),
                    0, 255).astype(np.uint8)
    return {"clahe": contrast_rgb, "green": contrast_rgb[:, :, 1], "frangi": vess_rgb, "final": final}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="cartella del dataset originale (HandInfo.csv + Hands/)")
    ap.add_argument("--images_dir", default=None, help="default: <dataset_dir>/Hands")
    ap.add_argument("--csv", default=None, help="default: <dataset_dir>/HandInfo.csv")
    ap.add_argument("--out_dir", default="./vein_pipeline_steps")
    ap.add_argument("--n", type=int, default=12, help="numero di immagini da salvare")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    images_dir = Path(args.images_dir) if args.images_dir else root / "Hands"
    csv_path = Path(args.csv) if args.csv else root / "HandInfo.csv"
    if not images_dir.is_dir():
        raise SystemExit(f"Cartella immagini non trovata: {images_dir.resolve()} (usa --images_dir)")
    if not csv_path.is_file():
        raise SystemExit(f"HandInfo.csv non trovato: {csv_path.resolve()} (usa --csv)")

    df = pd.read_csv(csv_path)
    for col in ("id", "imageName", "aspectOfHand"):
        if col not in df.columns:
            raise SystemExit(f"Colonna mancante nel CSV: {col}")
    df = df[df["aspectOfHand"].astype(str).str.contains("dorsal", case=False)]
    if df.empty:
        raise SystemExit("Nessuna immagine dorsale nel CSV.")

    rows = list(df[["id", "imageName", "aspectOfHand"]].itertuples(index=False, name=None))
    random.Random(args.seed).shuffle(rows)

    out = Path(args.out_dir)
    pre_dir = out / "preprocessed"
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for k, (subject_id, name, side) in enumerate(rows, start=1):
        if saved >= args.n:
            break
        img_path = images_dir / str(name)
        # preprocessing reale (stessa funzione di preProcessing.py, in-process)
        res = PP.process_single_image((img_path, subject_id, side, k, pre_dir))
        if res["status"] != "success":
            print(f"[{res['status'].upper()}] {name}: {res.get('reason')}")
            continue

        pre_png = pre_dir / str(subject_id) / f"{subject_id}_{side}_{k:03d}_dorsal_hand.png"
        pre_bgr = cv2.imread(str(pre_png))
        if pre_bgr is None:
            print(f"[WARN] output preprocessing non trovato: {pre_png}")
            continue
        pre = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB)             # come DorsalBiometricDataset._load_rgb
        raw = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        st = vein_stages(pre)
        final = enhance_veins(pre)
        if not np.array_equal(st["final"], final):
            print(f"[ATTENZIONE] stadi != enhance_veins su {name}: controlla i parametri")

        panel = np.hstack([label(fit_height(raw), "originale (grezza)"),
                           label(to_rgb3(pre), "preprocessing"),
                           label(to_rgb3(st["clahe"]), "CLAHE forte (L)"),
                           label(to_rgb3(st["green"]), "canale verde"),
                           label(to_rgb3(st["frangi"]), "Frangi"),
                           label(to_rgb3(final), "enhance_veins")])
        safe = Path(name).stem.replace(" ", "_")
        cv2.imwrite(str(out / f"{safe}_stages.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{safe}_enhanced.png"), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
        saved += 1

    print(f"Salvate {saved} immagini in {out.resolve()}")


if __name__ == "__main__":
    main()