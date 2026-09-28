"""
============================================================
PREPROCESSING PIPELINE - RICONOSCIMENTO BIOMETRICO MANO
Versione aggiornata: sfondo adattivo + detection multi-margine
============================================================
"""

from pathlib import Path
import os
import re
import cv2
import json
import math
import numpy as np

from concurrent.futures import ProcessPoolExecutor
import mediapipe as mp


# ============================================================
# CONFIGURAZIONE OUTPUT E SOGLIE
# ============================================================

# Dimensioni output per i vari rami
VIT_SIZE = (224, 224)
PALM_ROI_SIZE = (224, 224)
DORSAL_HAND_SIZE = (224, 224)
DORSAL_KNUCKLE_SIZE = (224, 224)
PALM_KNUCKLE_SIZE = (96, 96)

# Soglie di qualita' / confidenza (PERMISSIVE PER DETECT DIFFICILI)
MIN_HANDEDNESS_SCORE = 0.30  # Abbassato da 0.50 per catturare mani incerte
MIN_HAND_AREA_RATIO = 0.005  # Abbassato da 0.01 per piccoli crop
MIN_SHARPNESS = 1.5          # Varianza Laplaciano minima

# Sappiamo che OGNI immagine contiene una mano e che lato/vista arrivano dal
# nome file: handedness, area e sharpness NON devono far scartare il campione
# (un campione scartato viene poi perso dalla fusione). Di default sono
# "soft": generano solo un warning nel metadata. Mettere True per il vecchio
# comportamento (scarto).
STRICT_QUALITY_FILTERS = False

# Margini di sfondo provati in sequenza se la mano non viene rilevata.
# 0.12 = comportamento precedente (identico per le immagini gia' funzionanti).
PADDING_LADDER = (0.12, 0.30, 0.55)
MAX_UPSCALE_SIDE = 2400      # niente upscale 2x oltre questa dimensione

# Salva in <output_dir>/_failed_debug le immagini per cui la detection fallisce
SAVE_FAILED_DEBUG = True


# ============================================================
# MEDIAPIPE INIZIALIZZAZIONE
# ============================================================

mp_hands = mp.solutions.hands
HANDS_DETECTOR = None


def init_worker():
    """
    Crea una singola istanza MediaPipe per processo worker
    con soglia di confidenza abbassata a 0.05.
    """
    global HANDS_DETECTOR

    HANDS_DETECTOR = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.05  # Abbassato da 0.10
    )


# ============================================================
# CARICAMENTO IMMAGINE E SFONDO ADATTIVO
# ============================================================

BLACK_BORDER_RATIO = 0.12   # margine di default (primo tentativo)
BLACK_BORDER_MIN_PX = 30    # margine minimo in pixel


def read_image_bgr(img_path):
    """
    Legge l'immagine in BGR a 8 bit gestendo i casi che cv2.imread(IMREAD_COLOR)
    tratta male:
    - PNG con canale alpha (sfondo trasparente): IMREAD_COLOR scarta l'alpha e
      i pixel "trasparenti" tornano con un colore arbitrario (spesso rumore o
      bianco). Qui l'immagine viene composta su NERO usando l'alpha.
    - immagini a 16 bit / scala di grigi.
    - percorsi con caratteri non ASCII (imread ritorna None su Windows).
    """
    try:
        data = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None

    if img is None:
        return None

    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        return np.clip(bgr * alpha, 0, 255).astype(np.uint8)

    return img


def estimate_background_color(img):
    """
    Stima il colore di sfondo dai 4 ANGOLI (mediana), non da tutto il bordo:
    in un ritaglio stretto la mano/il polso toccano i lati, gli angoli invece
    sono quasi sempre sfondo. Per sfondo nero ritorna ~(0, 0, 0), quindi il
    comportamento resta identico a prima.
    """
    h, w = img.shape[:2]
    p = max(4, int(round(min(h, w) * 0.03)))
    p = min(p, h // 2, w // 2)
    corners = np.concatenate([
        img[:p, :p].reshape(-1, 3),
        img[:p, -p:].reshape(-1, 3),
        img[-p:, :p].reshape(-1, 3),
        img[-p:, -p:].reshape(-1, 3),
    ])
    med = np.median(corners, axis=0)
    return tuple(int(v) for v in med)


def pad_image(img, ratio, bg_color, min_border_px=BLACK_BORDER_MIN_PX):
    """Aggiunge un margine del colore di sfondo stimato (non piu' sempre nero)."""
    h, w = img.shape[:2]
    border_x = max(int(min_border_px), int(round(w * ratio)))
    border_y = max(int(min_border_px), int(round(h * ratio)))
    return cv2.copyMakeBorder(
        img,
        border_y, border_y, border_x, border_x,
        borderType=cv2.BORDER_CONSTANT,
        value=bg_color
    )


def load_image_with_padded_background(
    img_path,
    border_ratio=BLACK_BORDER_RATIO,
    min_border_px=BLACK_BORDER_MIN_PX
):
    """Compatibilita': legge + padding con sfondo stimato (singolo margine)."""
    img = read_image_bgr(img_path)
    if img is None:
        return None
    return pad_image(img, border_ratio, estimate_background_color(img), min_border_px)


# ============================================================
# UTILITY GEOMETRICHE ED ELABORAZIONE
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
    src_w, src_h = src_size
    target_w, target_h = target_size
    scale = min(target_w / src_w, target_h / src_h)
    return cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC


def resize_with_crop_fill(img, target_size=(224, 224), interpolation=None):
    if not valid_image(img):
        return None

    target_w, target_h = target_size
    h, w = img.shape[:2]

    if interpolation is None:
        interpolation = adaptive_interpolation((w, h), target_size)

    scale = max(target_w / w, target_h / h)

    new_w = max(target_w, int(round(w * scale)))
    new_h = max(target_h, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    x_offset = (new_w - target_w) // 2
    y_offset = (new_h - target_h) // 2

    return resized[y_offset:y_offset + target_h, x_offset:x_offset + target_w]


def compute_quality_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    return {
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2)
    }


def canonicalize_laterality(img, coords, hand_side):
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
# CROP E ROTAZIONE GEOMETRICA
# ============================================================

def crop_hand_from_landmarks(img, coords, padding_ratio=0.30, min_padding_px=20):
    h, w = img.shape[:2]

    min_x = float(coords[:, 0].min())
    max_x = float(coords[:, 0].max())
    min_y = float(coords[:, 1].min())
    max_y = float(coords[:, 1].max())

    hand_w = max(max_x - min_x, 20.0)
    hand_h = max(max_y - min_y, 20.0)

    pad_x = max(hand_w * padding_ratio, float(min_padding_px))
    pad_y = max(hand_h * padding_ratio, float(min_padding_px))

    requested_x1 = int(np.floor(min_x - pad_x))
    requested_y1 = int(np.floor(min_y - pad_y))
    requested_x2 = int(np.ceil(max_x + pad_x))
    requested_y2 = int(np.ceil(max_y + pad_y))

    x1 = max(0, requested_x1)
    y1 = max(0, requested_y1)
    x2 = min(w, requested_x2)
    y2 = min(h, requested_y2)

    if x2 <= x1 or y2 <= y1:
        return None, None, None, None

    hand_crop = img[y1:y2, x1:x2].copy()

    coords_local = coords.copy()
    coords_local[:, 0] -= x1
    coords_local[:, 1] -= y1

    bbox = [x1, y1, x2, y2]

    missing_padding = {
        "left": int(max(0, -requested_x1)),
        "top": int(max(0, -requested_y1)),
        "right": int(max(0, requested_x2 - w)),
        "bottom": int(max(0, requested_y2 - h))
    }

    return hand_crop, coords_local, bbox, missing_padding


def rotate_hand_upright(img, coords, bg_color=(0, 0, 0)):
    h, w = img.shape[:2]

    wrist = coords[0]
    middle_mcp = coords[9]

    dx = float(middle_mcp[0] - wrist[0])
    dy = float(middle_mcp[1] - wrist[1])

    length = math.sqrt(dx * dx + dy * dy)
    if length < 5:
        return None, None, None

    current_angle = math.degrees(math.atan2(dy, dx))
    rotation_angle = current_angle + 90.0

    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_color
    )

    rotated_coords = transform_points(coords, M)
    return rotated, rotated_coords, M


def crop_final_hand(img, coords, padding_ratio=0.10):
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


def normalize_hand_geometry(img, original_coords, initial_padding_ratio=0.30, final_padding_ratio=0.10,
                            bg_color=(0, 0, 0)):
    hand_crop, local_coords, original_bbox, missing_padding = crop_hand_from_landmarks(
        img, original_coords, padding_ratio=initial_padding_ratio
    )

    if not valid_image(hand_crop):
        return None, None, None, None

    rotated_img, rotated_coords, M = rotate_hand_upright(hand_crop, local_coords, bg_color=bg_color)

    if not valid_image(rotated_img):
        return None, None, None, None

    final_img, final_coords = crop_final_hand(
        rotated_img, rotated_coords, padding_ratio=final_padding_ratio
    )

    if not valid_image(final_img):
        return None, None, None, None

    return final_img, final_coords, original_bbox, missing_padding


# ============================================================
# ESTRAZIONE ROI (PALMO E NOCCHIE)
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


def extract_central_palm_roi(img, coords):
    wrist = coords[0]
    mcps = coords[[5, 9, 13, 17]]

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

    return safe_rect_crop(
        img,
        min_x - side_pad,
        top_y - top_pad,
        max_x + side_pad,
        bottom_y + bottom_pad
    )


KNUCKLE_PARAMS = {
    "mcp": {"width_ratio": 1.50, "height_ratio": 0.70, "min_width": 32, "min_height": 24},
    "pip": {"width_ratio": 1.35, "height_ratio": 0.65, "min_width": 28, "min_height": 22},
    "dip": {"width_ratio": 1.20, "height_ratio": 0.60, "min_width": 24, "min_height": 20}
}

FINGER_CHAINS = {
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20]
}


def _stable_direction(coords, prev_idx, curr_idx, next_idx):
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
    ref_idx_pair = (curr_idx, next_idx) if next_idx is not None else (prev_idx, curr_idx)
    ref_length = float(np.linalg.norm(coords[ref_idx_pair[1]] - coords[ref_idx_pair[0]]))

    return direction, ref_length


def extract_knuckle_roi(img, coords, prev_idx, joint_idx, next_idx, joint_type="mcp", bg_color=(0, 0, 0)):
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
        borderValue=bg_color
    )

    return crop if valid_image(crop) else None


def extract_all_knuckle_rois(img, coords, bg_color=(0, 0, 0)):
    rois = {}
    for finger_name, chain in FINGER_CHAINS.items():
        mcp_idx, pip_idx, dip_idx, tip_idx = chain
        joint_specs = [
            ("mcp", None, mcp_idx, pip_idx),
            ("pip", mcp_idx, pip_idx, dip_idx),
            ("dip", pip_idx, dip_idx, tip_idx)
        ]
        for joint_type, prev_idx, joint_idx, next_idx in joint_specs:
            roi = extract_knuckle_roi(img, coords, prev_idx, joint_idx, next_idx, joint_type=joint_type, bg_color=bg_color)
            rois[f"{finger_name}_{joint_type}"] = roi
    return rois


# ============================================================
# PREPROCESSING FOTOMETRICO
# ============================================================

def illumination_correction_gray(gray, sigma=25):
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.divide(gray, background, scale=128)


def preprocess_rgb(img, sigma=25, clahe_clip=1.8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = illumination_correction_gray(L, sigma=sigma)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    L = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)


def preprocess_knuckle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corrected = illumination_correction_gray(gray, sigma=15)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(corrected)


# ============================================================
# DETECTION MULTI-TENTATIVO
# ============================================================

def _detect_on(bgr_img):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    res = HANDS_DETECTOR.process(rgb_img)
    if res and res.multi_hand_landmarks:
        return res
    return None


def detect_hand_multi(raw_img, bg_color):
    """
    Prova a rilevare la mano variando margine di sfondo, contrasto, scala e
    rotazione. Il primo margine (0.12) con lo stesso ordine di varianti del
    codice precedente: le immagini che gia' funzionavano producono gli stessi
    landmark. Solo se falliscono si passa a margini piu' ampi (0.30, 0.55):
    MediaPipe (palm detector) va in difficolta' sia con mani troppo grandi nel
    frame / dita tagliate ai bordi (ritagli stretti), sia con sfondo
    innaturale rispetto al ritaglio.

    Ritorna (results, immagine_su_cui_valgono_i_landmark, tentativo) oppure
    (None, None, None).
    """
    for ratio in PADDING_LADDER:
        img = pad_image(raw_img, ratio, bg_color)
        h, w = img.shape[:2]
        tag = f"pad{int(round(ratio * 100))}"

        # A: CLAHE (i landmark valgono comunque su `img`, stessa geometria)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_det = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced_bgr = cv2.cvtColor(cv2.merge([clahe_det.apply(l), a, b]), cv2.COLOR_LAB2BGR)
        res = _detect_on(enhanced_bgr)
        if res:
            return res, img, f"{tag}/clahe"

        # B: RGB standard
        res = _detect_on(img)
        if res:
            return res, img, f"{tag}/rgb"

        # C: upscale 2x (mani piccole nel frame)
        if max(h, w) <= MAX_UPSCALE_SIDE:
            upscaled = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            res = _detect_on(upscaled)
            if res:
                return res, upscaled, f"{tag}/upscale2x"

        # D: equalizzazione istogramma globale
        gray_eq = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        res = _detect_on(cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR))
        if res:
            return res, img, f"{tag}/equalize"

        # E: piccole rotazioni (bordi riempiti col colore di sfondo, non nero)
        center = (w / 2, h / 2)
        for angle in (15, -15, 30, -30, 45, -45):
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M_rot, (w, h), borderValue=bg_color)
            res = _detect_on(rotated)
            if res:
                return res, rotated, f"{tag}/rot{angle}"

    return None, None, None


def _save_failed_debug(raw_img, bg_color, subject_id, img_path, output_dir):
    if not SAVE_FAILED_DEBUG:
        return
    try:
        dbg_dir = Path(output_dir) / "_failed_debug"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        name = f"{subject_id}_{Path(img_path).stem}_bg{bg_color[2]}-{bg_color[1]}-{bg_color[0]}.png"
        cv2.imwrite(str(dbg_dir / name), pad_image(raw_img, 0.30, bg_color))
    except Exception:
        pass


def _center_crop(img, frac=0.6):
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y1, x1 = (h - ch) // 2, (w - cw) // 2
    return img[y1:y1 + ch, x1:x1 + cw].copy()


# ============================================================
# PROCESSAMENTO SINGOLA IMMAGINE
# ============================================================

def process_single_image(task):
    global HANDS_DETECTOR
    (img_path, subject_id, hand_side, is_dorsal, seq_num, output_dir) = task

    try:
        # 1. Caricamento (gestisce alpha/16 bit) e stima dello sfondo reale
        raw = read_image_bgr(img_path)

        if raw is None:
            return {"status": "error", "file": str(img_path), "reason": "Immagine non trovata o corrotta"}

        bg_color = estimate_background_color(raw)

        if HANDS_DETECTOR is None:
            init_worker()

        # 2. Detection a tentativi multipli (margini di sfondo + varianti)
        results, img, attempt = detect_hand_multi(raw, bg_color)

        if results is None:
            _save_failed_debug(raw, bg_color, subject_id, img_path, output_dir)
            return {"status": "skipped", "file": str(img_path), "reason": "Nessuna mano rilevata"}

        h, w = img.shape[:2]
        landmarks = results.multi_hand_landmarks[0].landmark
        coords_original = landmarks_to_pixels(landmarks, w, h)
        warnings = []

        # 3. Controlli di qualita' (soft di default: il lato arriva dal nome
        # file, quindi la handedness del detector non serve a nulla di critico)
        handedness_score = 1.0
        if results.multi_handedness:
            handedness_score = float(results.multi_handedness[0].classification[0].score)
            if handedness_score < MIN_HANDEDNESS_SCORE:
                msg = f"Confidenza handedness bassa ({handedness_score:.2f})"
                if STRICT_QUALITY_FILTERS:
                    return {"status": "skipped", "file": str(img_path), "reason": msg}
                warnings.append(msg)

        bbox_w = float(coords_original[:, 0].max() - coords_original[:, 0].min())
        bbox_h = float(coords_original[:, 1].max() - coords_original[:, 1].min())
        area_ratio = (bbox_w * bbox_h) / float(w * h)

        if area_ratio < MIN_HAND_AREA_RATIO:
            msg = f"Bounding box mano troppo piccola ({area_ratio:.4f})"
            if STRICT_QUALITY_FILTERS:
                return {"status": "skipped", "file": str(img_path), "reason": msg}
            warnings.append(msg)

        # 4. Normalizzazione geometrica (rotazioni riempite con lo sfondo reale)
        img, coords_original, was_mirrored = canonicalize_laterality(img, coords_original, hand_side)

        final_padding = 0.15 if is_dorsal else 0.10
        hand_img, coords, original_bbox, missing_padding = normalize_hand_geometry(
            img,
            coords_original,
            initial_padding_ratio=0.30,
            final_padding_ratio=final_padding,
            bg_color=bg_color
        )

        if not valid_image(hand_img):
            return {"status": "error", "file": str(img_path), "reason": "Errore normalizzazione geometrica"}

        # 5. Metriche di qualita' (calcolate a dimensione fissa, comparabili)
        quality = compute_quality_metrics(resize_with_crop_fill(hand_img, target_size=VIT_SIZE))
        if quality["sharpness"] < MIN_SHARPNESS:
            msg = f"Immagine molto sfocata (sharpness={quality['sharpness']})"
            if STRICT_QUALITY_FILTERS:
                return {"status": "skipped", "file": str(img_path), "reason": msg}
            warnings.append(msg)

        # 6. Salvataggio Output
        out_folder = Path(output_dir) / str(subject_id)
        out_folder.mkdir(parents=True, exist_ok=True)

        side_code = "L" if str(hand_side).lower().startswith("l") else "R"
        view_code = "dorsal" if is_dorsal else "palmar"
        # seq_num puo' essere un intero (chiamata standalone) oppure una
        # stringa con il sample_id completo, tag di augmentazione incluso
        # (chiamata dal fusion script, es. "1_aug2"): finisce nel nome file
        # cosi' scatti diversi della stessa mano non si sovrascrivono.
        seq_str = f"{seq_num:03d}" if isinstance(seq_num, int) else str(seq_num)
        base_name = f"{subject_id}_{side_code}_{view_code}_{seq_str}"

        palm_roi_fallback = False

        if not is_dorsal:
            # PALMO
            palm_hand_processed = preprocess_rgb(hand_img, sigma=25, clahe_clip=1.5)
            palm_hand_processed = resize_with_crop_fill(palm_hand_processed, target_size=VIT_SIZE)
            if not save_image(out_folder / f"{base_name}_palm_hand.png", palm_hand_processed):
                return {"status": "error", "file": str(img_path), "reason": "Scrittura palm_hand fallita"}

            # Il fusion script richiede SEMPRE palm_roi.png (find_processed_base):
            # se manca, il campione viene scartato in silenzio a valle anche se
            # qui risulta "success". Se l'ROI dai landmark e' degenere, si usa
            # il ritaglio centrale della mano come ripiego.
            central_palm_roi = extract_central_palm_roi(hand_img, coords)
            if not valid_image(central_palm_roi):
                palm_roi_fallback = True
                central_palm_roi = _center_crop(hand_img, 0.6)
                warnings.append("palm_roi da fallback (ROI da landmark degenere)")

            palm_roi_processed = preprocess_rgb(central_palm_roi, sigma=20, clahe_clip=1.8)
            palm_roi_processed = resize_with_crop_fill(palm_roi_processed, target_size=PALM_ROI_SIZE)
            if not save_image(out_folder / f"{base_name}_palm_roi.png", palm_roi_processed):
                return {"status": "error", "file": str(img_path), "reason": "Scrittura palm_roi fallita"}

            knuckle_rois = extract_all_knuckle_rois(hand_img, coords, bg_color=bg_color)
            for roi_name, knuckle in knuckle_rois.items():
                if valid_image(knuckle):
                    knuckle_processed = preprocess_knuckle(knuckle)
                    knuckle_processed = resize_with_crop_fill(knuckle_processed, target_size=PALM_KNUCKLE_SIZE)
                    save_image(out_folder / f"{base_name}_palm_{roi_name}.png", knuckle_processed)
        else:
            # DORSO
            dorsal_hand_processed = preprocess_rgb(hand_img, sigma=25, clahe_clip=1.5)
            dorsal_hand_processed = resize_with_crop_fill(dorsal_hand_processed, target_size=DORSAL_HAND_SIZE)
            if not save_image(out_folder / f"{base_name}_dorsal_hand.png", dorsal_hand_processed):
                return {"status": "error", "file": str(img_path), "reason": "Scrittura dorsal_hand fallita"}

            knuckle_rois = extract_all_knuckle_rois(hand_img, coords, bg_color=bg_color)
            for roi_name, knuckle in knuckle_rois.items():
                if valid_image(knuckle):
                    knuckle_processed = preprocess_rgb(knuckle, sigma=15, clahe_clip=1.5)
                    knuckle_processed = resize_with_crop_fill(knuckle_processed, target_size=DORSAL_KNUCKLE_SIZE)
                    save_image(out_folder / f"{base_name}_dorsal_{roi_name}.png", knuckle_processed)

        # 7. METADATA
        metadata = {
            "subject_id": str(subject_id),
            "hand_side": str(hand_side),
            "source_file": str(img_path),
            "is_dorsal": is_dorsal,
            "mirrored_to_canonical": was_mirrored,
            "handedness_confidence": round(handedness_score, 4),
            "detection_attempt": attempt,
            "background_color_bgr": list(bg_color),
            "palm_roi_fallback": palm_roi_fallback,
            "warnings": warnings,
            "quality": quality,
            "original_hand_bbox": original_bbox,
            "landmarks_original": coords_original.tolist(),
            "landmarks_normalized": coords.tolist()
        }

        with open(out_folder / f"{base_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {"status": "success", "file": str(img_path), "attempt": attempt, "warnings": warnings}

    except Exception as e:
        return {"status": "error", "file": str(img_path), "reason": str(e)}


# ============================================================
# GESTIONE STRUTTURA DATASET E MAIN
# ============================================================

FILENAME_PATTERN = re.compile(
    r"(?P<side>[LR])_(?P<view>dorsal|palmar)",
    re.IGNORECASE
)


def discover_tasks_from_participants(participants_dir, output_dir):
    participants_dir = Path(participants_dir)
    if not participants_dir.is_dir():
        raise FileNotFoundError(f"Cartella non trovata: {participants_dir}")

    tasks = []
    subject_dirs = sorted(p for p in participants_dir.iterdir() if p.is_dir())

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        rgb_dir = subject_dir / "RGB"

        if not rgb_dir.is_dir():
            continue

        png_files = sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")))
        seq_num = 0

        for img_path in png_files:
            match = FILENAME_PATTERN.search(img_path.stem)
            if not match:
                continue

            side_code = match.group("side").upper()
            view_code = match.group("view").lower()

            hand_side = "left" if side_code == "L" else "right"
            is_dorsal = (view_code == "dorsal")

            seq_num += 1
            tasks.append((img_path, subject_id, hand_side, is_dorsal, seq_num, output_dir))

    return tasks


def run_async_preprocessing(participants_dir, output_dir, max_workers=4, chunksize=8):
    tasks = discover_tasks_from_participants(participants_dir, output_dir)
    print(f"Totale immagini trovate: {len(tasks)}")

    success = 0
    skipped = 0
    errors = 0
    skip_reasons = {}
    attempts = {}
    n_warn = 0

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        for result in executor.map(process_single_image, tasks, chunksize=chunksize):
            if result["status"] == "success":
                success += 1
                attempts[result.get("attempt")] = attempts.get(result.get("attempt"), 0) + 1
                if result.get("warnings"):
                    n_warn += 1
                    print(f"[WARN] {result['file']}: {'; '.join(result['warnings'])}")
            elif result["status"] == "skipped":
                skipped += 1
                skip_reasons[result["reason"]] = skip_reasons.get(result["reason"], 0) + 1
                print(f"[SKIPPED] {result['file']}: {result['reason']}")
            else:
                errors += 1
                print(f"[ERROR] {result['file']}: {result['reason']}")

    print("\n==============================")
    print(f"SUCCESS: {success}  (con warning: {n_warn})")
    for att, count in sorted(attempts.items(), key=lambda x: -x[1]):
        print(f"   - rilevata con {att}: {count}")
    print(f"SKIPPED: {skipped}")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"   - {reason}: {count}")
    print(f"ERRORS: {errors}")
    print("==============================")


if __name__ == "__main__":
    PARTICIPANTS_DIR = r"D:\Users\Patrizio\Desktop\Tesi\dataset_zenodo\Participants"
    OUTPUT_DIR = r"D:\Users\Patrizio\Desktop\Tesi\new_dataset_preprocessed_NO_AUGEMENTATION"
    NUM_WORKERS = os.cpu_count() or 4

    run_async_preprocessing(PARTICIPANTS_DIR, OUTPUT_DIR, max_workers=NUM_WORKERS, chunksize=8)