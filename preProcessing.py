"""
============================================================
PREPROCESSING PIPELINE - RICONOSCIMENTO BIOMETRICO MANO
Versione ottimizzata

Modifiche rispetto alla versione precedente:

  1. Mirroring canonico L/R: tutte le mani sinistre vengono
     flippate orizzontalmente prima della normalizzazione
     geometrica, cosi' tutti gli stream lavorano su una sola
     geometria "canonica" invece di doverne imparare due
     speculari (dimezza la varianza intra-classe di forma).

  2. Interpolazione adattiva: INTER_AREA per downscale,
     INTER_CUBIC per upscale. Le ROI nocche sono piccole
     (24-90px) e vengono ingrandite anche di 5-9x: usare
     sempre INTER_AREA (pensato per shrink) degradava la
     texture, dato critico per il canale FRIT.

  3. Controlli di qualita' post-detection: score di
     handedness, plausibilita' geometrica della bbox,
     nitidezza (varianza Laplaciano) e luminosita' media.
     Salvati in metadata per essere riusati dal modulo di
     quality-estimation della fusione cross-attention a
     valle, e usati qui per scartare detection inaffidabili.

  4. Preprocessing coerente per i backbone congelati: palm_hand
     (ViT-S/DINOv2) e dorsal_hand (Swin-Tiny) usano la stessa
     funzione simmetrica sui canali colore, senza boost mirato
     sul verde. Un boost asimmetrico sposta le statistiche di
     input lontano dal pretraining ImageNet, danneggiando le
     feature di un backbone che non puo' riadattarsi (pesi
     fissi). Il grayscale resta solo per l'unico stream davvero
     handcrafted (FRIT), non e' un'incoerenza.

  5. Direzione delle ROI nocche stabilizzata: quando
     disponibile, la direzione anatomica usata per orientare
     il rettangolo di crop e' la media tra il segmento
     entrante e quello uscente dall'articolazione, non solo
     quello uscente. Riduce la sensibilita' a piccoli errori
     di landmark su singole falangi.

  6. Knuckle ROI palmari ridimensionate a 96x96 (invece di
     128x128) per contenere il fattore di ingrandimento e
     ridurre gli artefatti di interpolazione che si
     propagano nella successiva estrazione FRIT.

  7. Esecuzione parallela con executor.map + chunksize
     invece di sottomettere tutti i Future in una lista,
     per ridurre overhead di IPC e memoria su dataset grandi.
============================================================
"""

from pathlib import Path
import os
import cv2
import json
import math
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor

import mediapipe as mp


# ============================================================
# CONFIGURAZIONE OUTPUT
# ============================================================

# Stream palmo - ViT-S / DINOv2 (texture + forma, long-range)
VIT_SIZE = (224, 224)

# Stream palmo ROI centrale - MobileNetV3-Large
PALM_ROI_SIZE = (224, 224)

# Stream dorso - Swin-Tiny (forma + vasi sanguigni)
DORSAL_HAND_SIZE = (224, 224)

# Stream nocche dorso - MobileNetV3-Large
DORSAL_KNUCKLE_SIZE = (224, 224)

# Stream nocche palmo - FRIT / ridgelet handcrafted
PALM_KNUCKLE_SIZE = (96, 96)

# Soglie di qualita' / confidenza
MIN_HANDEDNESS_SCORE = 0.75
MIN_HAND_AREA_RATIO = 0.02   # bbox mano / area immagine
MIN_SHARPNESS = 15.0         # varianza Laplaciano, sotto = troppo sfocata


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

HANDS_DETECTOR = None


def init_worker():
    """
    Crea una singola istanza MediaPipe per processo worker.
    """

    global HANDS_DETECTOR

    HANDS_DETECTOR = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6
    )


# ============================================================
# UTILITY
# ============================================================

def valid_image(img):

    return (
        img is not None
        and isinstance(img, np.ndarray)
        and img.size > 0
        and img.shape[0] > 2
        and img.shape[1] > 2
    )


def landmarks_to_pixels(landmarks, w, h):

    return np.array(
        [[lm.x * w, lm.y * h] for lm in landmarks],
        dtype=np.float32
    )


def transform_points(points, M):

    points = np.asarray(points, dtype=np.float32)
    ones = np.ones((len(points), 1), dtype=np.float32)
    homogeneous = np.concatenate([points, ones], axis=1)

    return homogeneous @ M.T


def save_image(path, img):

    if not valid_image(img):
        return False

    return cv2.imwrite(str(path), img)


def adaptive_interpolation(src_size, target_size):
    """
    Sceglie l'interpolazione in base alla direzione del resize.

    INTER_AREA e' ottimale per rimpicciolire (moire-free),
    ma su un ingrandimento produce risultati poco piu' di un
    nearest neighbour. INTER_CUBIC e' molto migliore quando
    si ingrandisce (es. ROI nocche 30px -> 96/224px), essenziale
    per non introdurre artefatti ad alta frequenza che
    inquinerebbero il successivo calcolo FRIT.
    """

    src_w, src_h = src_size
    target_w, target_h = target_size

    scale = min(target_w / src_w, target_h / src_h)

    return cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC


def resize_with_padding(img, target_size=(224, 224), interpolation=None):
    """
    Ridimensiona l'immagine mantenendo l'aspect ratio.
    L'immagine viene inserita in una canvas della dimensione
    richiesta (letterbox).
    """

    if not valid_image(img):
        return None

    target_w, target_h = target_size
    h, w = img.shape[:2]

    if interpolation is None:
        interpolation = adaptive_interpolation((w, h), target_size)

    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    if len(img.shape) == 2:
        canvas = np.zeros((target_h, target_w), dtype=img.dtype)
    else:
        canvas = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


# ============================================================
# QUALITA' IMMAGINE
# ============================================================

def compute_quality_metrics(img):
    """
    Metriche di qualita' leggere calcolate una sola volta
    sulla mano gia' ritagliata (non sull'immagine originale
    intera, per non essere influenzate dallo sfondo).

    Vengono salvate nel metadata e possono essere usate:

      - qui, per scartare immagini palesemente inutilizzabili
      - a valle, come prior per il quality-estimator della
        fusione cross-attention (vedi architettura multi-stream)
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    return {
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2)
    }


# ============================================================
# CANONICALIZZAZIONE LATERALITA'
# ============================================================

def canonicalize_laterality(img, coords, hand_side):
    """
    Flippa orizzontalmente le mani sinistre in modo che ogni
    stream lavori sempre sulla stessa geometria canonica
    (equivalente a una mano destra).

    Senza questo passaggio i backbone shape/texture devono
    imparare due geometrie speculari, aumentando inutilmente
    la varianza intra-classe che il modello deve assorbire.
    """

    side_string = str(hand_side).lower()
    is_left = "left" in side_string or "sinistr" in side_string

    if not is_left:
        return img, coords, False

    w = img.shape[1]

    flipped_img = cv2.flip(img, 1)

    flipped_coords = coords.copy()
    flipped_coords[:, 0] = w - flipped_coords[:, 0]

    return flipped_img, flipped_coords, True


# ============================================================
# STEP 1 - CROP DELLA MANO ORIGINALE
# ============================================================

def crop_hand_from_landmarks(img, coords, padding_ratio=0.25):
    """
    Ritaglia la mano dall'immagine originale. Il padding viene
    mantenuto abbastanza grande per evitare di perdere parti
    anatomiche durante la successiva rotazione.
    """

    h, w = img.shape[:2]

    min_x = float(coords[:, 0].min())
    max_x = float(coords[:, 0].max())
    min_y = float(coords[:, 1].min())
    max_y = float(coords[:, 1].max())

    hand_w = max(max_x - min_x, 20)
    hand_h = max(max_y - min_y, 20)

    pad_x = hand_w * padding_ratio
    pad_y = hand_h * padding_ratio

    x1 = max(0, int(np.floor(min_x - pad_x)))
    y1 = max(0, int(np.floor(min_y - pad_y)))
    x2 = min(w, int(np.ceil(max_x + pad_x)))
    y2 = min(h, int(np.ceil(max_y + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    hand_crop = img[y1:y2, x1:x2].copy()

    coords_local = coords.copy()
    coords_local[:, 0] -= x1
    coords_local[:, 1] -= y1

    bbox = [x1, y1, x2, y2]

    return hand_crop, coords_local, bbox


# ============================================================
# STEP 2 - PADDING SU CANVAS QUADRATA
# ============================================================

def pad_to_large_square(img, coords, padding_ratio=0.45):
    """
    Inserisce la mano in una canvas quadrata piu' grande. Il
    padding aggiuntivo riduce il rischio di tagliare dita o
    polso durante la rotazione.
    """

    h, w = img.shape[:2]
    side = max(h, w)
    extra = int(side * padding_ratio)
    canvas_size = side + 2 * extra

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=img.dtype)

    x_offset = (canvas_size - w) // 2
    y_offset = (canvas_size - h) // 2

    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img

    coords_padded = coords.copy()
    coords_padded[:, 0] += x_offset
    coords_padded[:, 1] += y_offset

    return canvas, coords_padded


# ============================================================
# STEP 3 - ROTAZIONE DELLA MANO
# ============================================================

def rotate_hand_upright(img, coords):
    """
    Allinea la mano usando l'asse Wrist(0) -> Middle MCP(9).
    Dopo la rotazione il dito medio punta verso l'alto.
    """
    h, w = img.shape[:2]

    wrist = coords[0]
    middle_mcp = coords[9]

    dx = float(middle_mcp[0] - wrist[0])
    dy = float(middle_mcp[1] - wrist[1])

    length = math.sqrt(dx * dx + dy * dy)

    if length < 5:
        return None, None, None

    current_angle = math.degrees(math.atan2(dy, dx))
    
    # FORMARA CORRETTA: porta qualsiasi inclinazione a -90 gradi (verticale)
    rotation_angle = current_angle + 90.0

    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    rotated_coords = transform_points(coords, M)

    return rotated, rotated_coords, M


# ============================================================
# CROP FINALE DELLA MANO
# ============================================================

def crop_final_hand(img, coords, padding_ratio=0.15):
    """
    Elimina il grande padding introdotto prima della rotazione.
    """

    h, w = img.shape[:2]

    min_x = coords[:, 0].min()
    max_x = coords[:, 0].max()
    min_y = coords[:, 1].min()
    max_y = coords[:, 1].max()

    box_w = max_x - min_x
    box_h = max_y - min_y

    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    x1 = max(0, int(np.floor(min_x - pad_x)))
    y1 = max(0, int(np.floor(min_y - pad_y)))
    x2 = min(w, int(np.ceil(max_x + pad_x)))
    y2 = min(h, int(np.ceil(max_y + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = img[y1:y2, x1:x2].copy()

    new_coords = coords.copy()
    new_coords[:, 0] -= x1
    new_coords[:, 1] -= y1

    return crop, new_coords


# ============================================================
# PIPELINE GEOMETRICA COMPLETA
# ============================================================

def normalize_hand_geometry(
    img,
    original_coords,
    initial_padding_ratio=0.25,
    square_padding_ratio=0.45,
    final_padding_ratio=0.15
):
    """
    Pipeline: crop originale -> padding quadrato -> rotazione
    -> crop finale.
    """

    hand_crop, local_coords, original_bbox = crop_hand_from_landmarks(
        img, original_coords, padding_ratio=initial_padding_ratio
    )

    if not valid_image(hand_crop):
        return None, None, None

    padded_img, padded_coords = pad_to_large_square(
        hand_crop, local_coords, padding_ratio=square_padding_ratio
    )

    rotated_img, rotated_coords, M = rotate_hand_upright(padded_img, padded_coords)

    if not valid_image(rotated_img):
        return None, None, None

    final_img, final_coords = crop_final_hand(
        rotated_img, rotated_coords, padding_ratio=final_padding_ratio
    )

    if not valid_image(final_img):
        return None, None, None

    return final_img, final_coords, original_bbox


# ============================================================
# SAFE RECTANGLE
# ============================================================

def safe_rect_crop(img, x1, y1, x2, y2):

    h, w = img.shape[:2]

    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(1, min(w, x2)))
    y2 = int(max(1, min(h, y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2].copy()

    return crop if valid_image(crop) else None


# ============================================================
# ROI CENTRALE DEL PALMO
# ============================================================

def extract_central_palm_roi(img, coords):
    """
    Estrae la regione centrale del palmo usando Wrist(0) e
    MCP(5,9,13,17). Axis-aligned: valido perche' a questo punto
    della pipeline la mano e' gia' stata ruotata in verticale.
    """

    wrist = coords[0]
    mcp_indices = [5, 9, 13, 17]
    mcps = coords[mcp_indices]

    min_x = float(mcps[:, 0].min())
    max_x = float(mcps[:, 0].max())
    mcp_center_y = float(mcps[:, 1].mean())
    wrist_y = float(wrist[1])

    top_y = min(mcp_center_y, wrist_y)
    bottom_y = max(mcp_center_y, wrist_y)

    roi_width = max_x - min_x
    roi_height = bottom_y - top_y

    if roi_width < 10 or roi_height < 10:
        return None

    side_pad = roi_width * 0.15
    top_pad = roi_height * 0.12
    bottom_pad = roi_height * 0.12

    x1 = min_x - side_pad
    x2 = max_x + side_pad
    y1 = top_y - top_pad
    y2 = bottom_y + bottom_pad

    return safe_rect_crop(img, x1, y1, x2, y2)


# ============================================================
# PARAMETRI ROI NOCCHIE
# ============================================================

KNUCKLE_PARAMS = {
    "mcp": {"width_ratio": 1.50, "height_ratio": 0.70, "min_width": 32, "min_height": 24},
    "pip": {"width_ratio": 1.35, "height_ratio": 0.65, "min_width": 28, "min_height": 22},
    "dip": {"width_ratio": 1.20, "height_ratio": 0.60, "min_width": 24, "min_height": 20}
}

# Catena di landmark per dito: (mcp, pip, dip, tip)
FINGER_CHAINS = {
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20]
}


def _stable_direction(coords, prev_idx, curr_idx, next_idx):
    """
    Direzione anatomica usata per orientare la ROI attorno
    all'articolazione curr_idx.

    Se disponibile sia il segmento entrante (prev->curr) sia
    quello uscente (curr->next), la direzione e' la media dei
    due, normalizzata: riduce la sensibilita' a piccoli errori
    di landmark su una singola falange rispetto a usare un solo
    segmento (comportamento della versione precedente).
    """

    curr = coords[curr_idx]
    directions = []

    if prev_idx is not None:
        seg = curr - coords[prev_idx]
        norm = np.linalg.norm(seg)
        if norm >= 5:
            directions.append(seg / norm)

    if next_idx is not None:
        seg = coords[next_idx] - curr
        norm = np.linalg.norm(seg)
        if norm >= 5:
            directions.append(seg / norm)

    if not directions:
        return None, 0.0

    direction = np.mean(directions, axis=0)
    length = float(np.linalg.norm(direction))

    if length < 1e-6:
        return None, 0.0

    direction = direction / length

    # lunghezza di riferimento per dimensionare la ROI:
    # usa il segmento uscente se c'e', altrimenti l'entrante
    ref_idx_pair = (curr_idx, next_idx) if next_idx is not None else (prev_idx, curr_idx)
    ref_length = float(np.linalg.norm(coords[ref_idx_pair[1]] - coords[ref_idx_pair[0]]))

    return direction, ref_length


# ============================================================
# ROI NOCCHIA GENERICA (con direzione stabilizzata)
# ============================================================

def extract_knuckle_roi(img, coords, prev_idx, joint_idx, next_idx, joint_type="mcp"):
    """
    Estrae una ROI orientata attorno a un'articolazione,
    usando la direzione anatomica mediata tra segmento
    entrante e uscente quando entrambi disponibili.
    """

    if joint_type not in KNUCKLE_PARAMS:
        return None

    params = KNUCKLE_PARAMS[joint_type]

    direction, finger_length = _stable_direction(coords, prev_idx, joint_idx, next_idx)

    if direction is None or finger_length < 5:
        return None

    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
    center = coords[joint_idx].copy()

    roi_width = max(params["min_width"], int(finger_length * params["width_ratio"]))
    roi_height = max(params["min_height"], int(finger_length * params["height_ratio"]))

    half_w = roi_width / 2.0
    half_h = roi_height / 2.0

    top_left = center - perpendicular * half_w - direction * half_h
    top_right = center + perpendicular * half_w - direction * half_h
    bottom_right = center + perpendicular * half_w + direction * half_h
    bottom_left = center - perpendicular * half_w + direction * half_h

    src = np.float32([top_left, top_right, bottom_right, bottom_left])

    dst = np.float32([
        [0, 0],
        [roi_width - 1, 0],
        [roi_width - 1, roi_height - 1],
        [0, roi_height - 1]
    ])

    h, w = img.shape[:2]

    valid_points = np.logical_and.reduce([
        src[:, 0] >= 0, src[:, 0] < w,
        src[:, 1] >= 0, src[:, 1] < h
    ])

    if np.sum(valid_points) < 3:
        return None

    H = cv2.getPerspectiveTransform(src, dst)

    crop = cv2.warpPerspective(
        img, H, (roi_width, roi_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return crop if valid_image(crop) else None


# ============================================================
# ESTRAZIONE DI TUTTE LE ROI DELLE NOCCHIE (12 per lato)
# ============================================================

def extract_all_knuckle_rois(img, coords):
    """
    Estrae tutte le nocche di tutte le dita:
    index/middle/ring/pinky x mcp/pip/dip = 12 ROI.

    Per ciascuna articolazione la direzione e' stabilizzata
    mediando il segmento entrante e quello uscente della catena
    (vedi _stable_direction).
    """

    rois = {}

    for finger_name, chain in FINGER_CHAINS.items():

        mcp_idx, pip_idx, dip_idx, tip_idx = chain

        joint_specs = [
            ("mcp", None, mcp_idx, pip_idx),
            ("pip", mcp_idx, pip_idx, dip_idx),
            ("dip", pip_idx, dip_idx, tip_idx)
        ]

        for joint_type, prev_idx, joint_idx, next_idx in joint_specs:

            roi = extract_knuckle_roi(
                img, coords, prev_idx, joint_idx, next_idx,
                joint_type=joint_type
            )

            rois[f"{finger_name}_{joint_type}"] = roi

    return rois


# ============================================================
# PREPROCESSING FOTOMETRICO
# ============================================================

def illumination_correction_gray(gray, sigma=25):
    """
    Correzione dell'illuminazione tramite divisione per
    background a bassa frequenza.
    """

    background = cv2.GaussianBlur(gray, (0, 0), sigma)

    return cv2.divide(gray, background, scale=128)


def preprocess_rgb(img, sigma=25, clahe_clip=1.8):
    """
    Correzione illuminazione + CLAHE leggero sul canale L (Lab).
    """

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    L = illumination_correction_gray(L, sigma=sigma)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    L = clahe.apply(L)

    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)


def preprocess_knuckle(img):
    """
    Preprocessing delle nocche palmari (per FRIT):
    grayscale -> correzione illuminazione -> CLAHE.
    Pensato per preservare texture e struttura locale, non
    per l'estetica.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corrected = illumination_correction_gray(gray, sigma=15)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    return clahe.apply(corrected)


# NOTA IMPORTANTE su palm_hand / dorsal_hand:
#
# ViT-S/DINOv2 e Swin-Tiny sono backbone CONGELATI (pesi
# pretrained fissi). Un backbone frozen si aspetta input con
# statistiche colore vicine a quelle del suo pretraining
# (ImageNet): qualunque manipolazione asimmetrica per-canale
# (es. boost aggressivo solo sul verde per "far risaltare le
# vene") sposta la distribuzione dei pixel lontano da quella
# di training, e le feature estratte peggiorano — senza che i
# pesi possano adattarsi per compensare.
#
# Per questo palm_hand e dorsal_hand usano la STESSA funzione
# preprocess_rgb, leggera e simmetrica sui canali colore
# (illuminazione + CLAHE mite solo sulla luminanza L in Lab).
# Il contrasto locale aumentato da CLAHE e' gia' sufficiente a
# rendere piu' visibili i pattern vascolari senza distorcere il
# bilanciamento cromatico atteso dal backbone.
#
# Se in futuro serve un canale vene dedicato, va costruito come
# input AGGIUNTIVO per un ramo allenabile separato (es. una
# piccola CNN su una vesselness map stile Frangi filter), MAI
# alterando l'input del backbone frozen.


# ============================================================
# PROCESSAMENTO SINGOLA IMMAGINE
# ============================================================

def process_single_image(task):

    global HANDS_DETECTOR

    (img_path, subject_id, hand_side, seq_num, output_dir) = task

    try:
        # --------------------------------------------------
        # LETTURA
        # --------------------------------------------------

        img = cv2.imread(str(img_path))

        if img is None:
            return {"status": "error", "file": str(img_path), "reason": "Immagine non trovata o corrotta"}

        h, w = img.shape[:2]

        # --------------------------------------------------
        # MEDIAPIPE
        # --------------------------------------------------

        if HANDS_DETECTOR is None:
            init_worker()

        results = HANDS_DETECTOR.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            return {"status": "skipped", "file": str(img_path), "reason": "Nessuna mano rilevata"}

        landmarks = results.multi_hand_landmarks[0].landmark
        coords_original = landmarks_to_pixels(landmarks, w, h)

        # --------------------------------------------------
        # CONTROLLO CONFIDENZA HANDEDNESS
        # --------------------------------------------------

        handedness_score = 1.0

        if results.multi_handedness:
            handedness_score = float(
                results.multi_handedness[0].classification[0].score
            )

            if handedness_score < MIN_HANDEDNESS_SCORE:
                return {
                    "status": "skipped",
                    "file": str(img_path),
                    "reason": f"Confidenza handedness bassa ({handedness_score:.2f})"
                }

        # --------------------------------------------------
        # CONTROLLO PLAUSIBILITA' GEOMETRICA
        # --------------------------------------------------

        bbox_w = float(coords_original[:, 0].max() - coords_original[:, 0].min())
        bbox_h = float(coords_original[:, 1].max() - coords_original[:, 1].min())
        area_ratio = (bbox_w * bbox_h) / float(w * h)

        if area_ratio < MIN_HAND_AREA_RATIO:
            return {
                "status": "skipped",
                "file": str(img_path),
                "reason": f"Bounding box mano troppo piccola ({area_ratio:.4f})"
            }

        # --------------------------------------------------
        # IDENTIFICAZIONE PALMO / DORSO
        # --------------------------------------------------

        side_string = str(hand_side).lower()
        is_dorsal = "dorsal" in side_string or "dorso" in side_string

        # --------------------------------------------------
        # CANONICALIZZAZIONE LATERALITA' (mirror mani sinistre)
        # --------------------------------------------------

        img, coords_original, was_mirrored = canonicalize_laterality(
            img, coords_original, hand_side
        )

        # --------------------------------------------------
        # PADDING DIFFERENZIATO E NORMALIZZAZIONE GEOMETRICA
        # --------------------------------------------------

        final_padding = 0.22 if is_dorsal else 0.14

        hand_img, coords, original_bbox = normalize_hand_geometry(
            img, coords_original,
            initial_padding_ratio=0.25,
            square_padding_ratio=0.45,
            final_padding_ratio=final_padding
        )

        if not valid_image(hand_img):
            return {"status": "error", "file": str(img_path), "reason": "Errore normalizzazione geometrica"}

        # --------------------------------------------------
        # METRICHE DI QUALITA' (post-crop, prima di qualunque
        # enhancement fotometrico, cosi' riflettono l'immagine
        # reale acquisita)
        # --------------------------------------------------

        quality = compute_quality_metrics(hand_img)

        if quality["sharpness"] < MIN_SHARPNESS:
            return {
                "status": "skipped",
                "file": str(img_path),
                "reason": f"Immagine troppo sfocata (sharpness={quality['sharpness']})"
            }

        # --------------------------------------------------
        # CARTELLA OUTPUT
        # --------------------------------------------------

        out_folder = Path(output_dir) / str(subject_id)
        out_folder.mkdir(parents=True, exist_ok=True)

        base_name = f"{subject_id}_{hand_side}_{seq_num:03d}"

        # ====================================================
        # PALMO
        # ====================================================

        if not is_dorsal:

            # Mano completa -> ViT-S / DINOv2 (forma + texture)
            palm_hand_processed = preprocess_rgb(hand_img, sigma=25, clahe_clip=1.5)
            palm_hand_processed = resize_with_padding(palm_hand_processed, target_size=VIT_SIZE)
            save_image(out_folder / f"{base_name}_palm_hand.png", palm_hand_processed)

            # ROI centrale palmo -> MobileNetV3-Large
            central_palm_roi = extract_central_palm_roi(hand_img, coords)

            if not valid_image(central_palm_roi):
                return {"status": "error", "file": str(img_path), "reason": "ROI centrale del palmo non valida"}

            palm_roi_processed = preprocess_rgb(central_palm_roi, sigma=20, clahe_clip=1.8)
            palm_roi_processed = resize_with_padding(palm_roi_processed, target_size=PALM_ROI_SIZE)
            save_image(out_folder / f"{base_name}_palm_roi.png", palm_roi_processed)

            # 12 ROI nocche -> FRIT / Ridgelet (handcrafted)
            knuckle_rois = extract_all_knuckle_rois(hand_img, coords)

            for roi_name, knuckle in knuckle_rois.items():

                if not valid_image(knuckle):
                    continue

                knuckle_processed = preprocess_knuckle(knuckle)
                knuckle_processed = resize_with_padding(knuckle_processed, target_size=PALM_KNUCKLE_SIZE)
                save_image(out_folder / f"{base_name}_palm_{roi_name}.png", knuckle_processed)

        # ====================================================
        # DORSO
        # ====================================================

        else:

            # Mano dorsale completa -> Swin-Tiny (frozen).
            # Stessa funzione usata per palm_hand: preserva le
            # statistiche colore attese dal pretraining ImageNet.
            dorsal_hand_processed = preprocess_rgb(hand_img, sigma=25, clahe_clip=1.5)
            dorsal_hand_processed = resize_with_padding(dorsal_hand_processed, target_size=DORSAL_HAND_SIZE)
            save_image(out_folder / f"{base_name}_dorsal_hand.png", dorsal_hand_processed)

            # 12 ROI nocche dorsali -> MobileNetV3-Large
            knuckle_rois = extract_all_knuckle_rois(hand_img, coords)

            for roi_name, knuckle in knuckle_rois.items():

                if not valid_image(knuckle):
                    continue

                knuckle_processed = preprocess_rgb(knuckle, sigma=15, clahe_clip=1.5)
                knuckle_processed = resize_with_padding(knuckle_processed, target_size=DORSAL_KNUCKLE_SIZE)
                save_image(out_folder / f"{base_name}_dorsal_{roi_name}.png", knuckle_processed)

        # ====================================================
        # METADATA
        # ====================================================

        metadata = {
            "subject_id": str(subject_id),
            "hand_side": str(hand_side),
            "source_file": str(img_path),
            "is_dorsal": is_dorsal,
            "mirrored_to_canonical": was_mirrored,
            "handedness_confidence": round(handedness_score, 4),
            "quality": quality,
            "original_hand_bbox": original_bbox,
            "landmarks_original": coords_original.tolist(),
            "landmarks_normalized": coords.tolist(),
            "output_sizes": {
                "palm_hand": list(VIT_SIZE),
                "palm_roi": list(PALM_ROI_SIZE),
                "palm_knuckle": list(PALM_KNUCKLE_SIZE),
                "dorsal_hand": list(DORSAL_HAND_SIZE),
                "dorsal_knuckle": list(DORSAL_KNUCKLE_SIZE)
            },
            "knuckle_structure": {
                finger: ["mcp", "pip", "dip"] for finger in FINGER_CHAINS
            }
        }

        with open(out_folder / f"{base_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {"status": "success", "file": str(img_path)}

    except Exception as e:

        return {"status": "error", "file": str(img_path), "reason": str(e)}


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def run_async_preprocessing(csv_path, images_dir, output_dir, max_workers=4, chunksize=8):

    # --------------------------------------------------------
    # LETTURA CSV
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    col_id = "id"
    col_img = "imageName"
    col_side = "aspectOfHand"

    for col in [col_id, col_img, col_side]:
        if col not in df.columns:
            raise ValueError(f"Colonna mancante: {col}")

    # --------------------------------------------------------
    # CREAZIONE TASK
    # --------------------------------------------------------

    tasks = []

    for subject_id, group in df.groupby(col_id, sort=False):

        for seq_num, (_, row) in enumerate(group.iterrows(), start=1):

            img_path = Path(images_dir) / str(row[col_img])

            tasks.append((img_path, subject_id, row[col_side], seq_num, output_dir))

    print(f"Totale immagini: {len(tasks)}")

    # --------------------------------------------------------
    # STATISTICHE
    # --------------------------------------------------------

    success = 0
    skipped = 0
    errors = 0
    skip_reasons = {}

    # --------------------------------------------------------
    # ELABORAZIONE PARALLELA
    # executor.map + chunksize: meno overhead di IPC rispetto a
    # sottomettere un Future per ogni singolo task, e non tiene
    # in memoria l'intera lista di Future contemporaneamente.
    # --------------------------------------------------------

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:

        for result in executor.map(process_single_image, tasks, chunksize=chunksize):

            if result["status"] == "success":
                success += 1

            elif result["status"] == "skipped":
                skipped += 1
                skip_reasons[result["reason"]] = skip_reasons.get(result["reason"], 0) + 1
                print(f"[SKIPPED] {result['file']}: {result['reason']}")

            else:
                errors += 1
                print(f"[ERROR] {result['file']}: {result['reason']}")

    # --------------------------------------------------------
    # RISULTATI FINALI
    # --------------------------------------------------------

    print("\n==============================")
    print(f"SUCCESS: {success}")
    print(f"SKIPPED: {skipped}")

    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"   - {reason}: {count}")

    print(f"ERRORS: {errors}")
    print("==============================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    CSV_FILE = r"C:\Users\Admin\Desktop\Tesi\Dataset\11k Hands dataset\HandInfo-test.csv"
    IMAGES_DIR = r"C:\Users\Admin\Desktop\Tesi\Dataset\11k Hands dataset\Hands-test"
    OUTPUT_DIR = "./dataset_preprocessed"
    NUM_WORKERS = os.cpu_count()

    run_async_preprocessing(CSV_FILE, IMAGES_DIR, OUTPUT_DIR, max_workers=NUM_WORKERS, chunksize=8)