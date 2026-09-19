"""
Debug rapido: prova diverse combinazioni su UNA immagine e
salva su disco l'immagine paddata, per capire perche' MediaPipe
non trova la mano.

Uso:
    python debug_single_image.py "D:\...\P002\RGB\P002_L_dorsal_processed.png"
"""

import sys
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands


def load_image_with_white_background(img_path, border_ratio=0.55, min_border_px=60):

    # Si legge sempre a 3 canali: l'alpha, quando presente in
    # questo dataset, non e' una maschera di trasparenza
    # affidabile e produceva un composito tutto bianco.
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if img is None:
        print("!! cv2.imread ha restituito None -> path errato o file corrotto")
        return None

    print(f"img.shape = {img.shape}, dtype = {img.dtype}")

    h, w = img.shape[:2]
    border_x = max(int(min_border_px), int(round(w * border_ratio)))
    border_y = max(int(min_border_px), int(round(h * border_ratio)))

    padded = cv2.copyMakeBorder(
        img, border_y, border_y, border_x, border_x,
        borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    return padded


def try_detect(img, label, model_complexity, min_conf):

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=model_complexity,
        min_detection_confidence=min_conf
    ) as detector:

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        found = bool(results.multi_hand_landmarks)
        print(f"[{label}] model_complexity={model_complexity} min_conf={min_conf} -> mano trovata: {found}")

        return found


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Uso: python debug_single_image.py <path_immagine.png>")
        sys.exit(1)

    path = sys.argv[1]

    padded = load_image_with_white_background(path)

    if padded is None:
        sys.exit(1)

    print(f"padded.shape = {padded.shape}")
    cv2.imwrite("debug_padded.png", padded)
    print("-> salvata debug_padded.png, apri e guarda se la mano si vede intera e su sfondo bianco")

    # originale, senza alcun padding/composito
    raw = cv2.imread(path, cv2.IMREAD_COLOR)
    if raw is not None:
        cv2.imwrite("debug_raw_no_alpha.png", raw)

    print()
    print("=== TEST DETECTION ===")

    # su immagine paddata, varie soglie/complessita'
    for mc in (0, 1):
        for conf in (0.5, 0.1, 0.02):
            try_detect(padded, "padded", mc, conf)

    # su immagine originale (senza padding bianco), per confronto
    if raw is not None:
        for mc in (0, 1):
            try_detect(raw, "raw", mc, 0.1)

    # su versione ridimensionata (in caso il problema sia la risoluzione)
    h, w = padded.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        small = cv2.resize(padded, (int(w * scale), int(h * scale)))
        print(f"Test anche su versione ridotta a {small.shape[1]}x{small.shape[0]}")
        try_detect(small, "padded_resized", 1, 0.1)