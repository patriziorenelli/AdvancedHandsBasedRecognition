"""
============================================================
PALM_CORE - Config + FRIT + Modelli + Dataset + Loss/Metriche
============================================================
Pipeline di embedding biometrico per lo stream PALMO, coerente
con l'output di preProcessing.py:

  <output_dir>/<subject_id>/<subject>_<side>_<seq>_palm_hand.png     -> ViT-S/DINOv2 (frozen) + MLP
  <output_dir>/<subject_id>/<subject>_<side>_<seq>_palm_roi.png      -> MobileNetV3-Large
  <output_dir>/<subject_id>/<subject>_<side>_<seq>_palm_<f>_<j>.png  -> canali FRIT (handcrafted)
  <output_dir>/<subject_id>/<subject>_<side>_<seq>_metadata.json

Contenuto:
  1. CONFIG
  2. FRIT (Finite Ridgelet Transform, handcrafted, no pesi appresi)
  3. MODELLI: TextureBranch / CentralROIBranch / KnuckleFRITBranch
     + PalmEmbeddingNet (fusione) + ArcMarginHead (solo training)
  4. DATASET: legge direttamente l'output di preProcessing.py
  5. LOSS / METRICHE: ArcFace+CE, triplet opzionale, EER open-set
============================================================
"""

from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from skimage.transform import radon, iradon
import timm


# ============================================================
# 1) CONFIG
# ============================================================
class Config:
    DATA_DIR = Path("./dataset_preprocessed")     # output di preProcessing.py
    CHECKPOINT_DIR = Path("./checkpoints")
    FINAL_MODEL_DIR = Path("./models_final")

    VIT_SIZE = (224, 224)
    PALM_ROI_SIZE = (224, 224)
    PALM_KNUCKLE_SIZE = (96, 96)

    EMBEDDING_DIM = 256
    TEXTURE_EMBED_DIM = 128
    ROI_EMBED_DIM = 128
    KNUCKLE_EMBED_DIM = 128

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 60
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    ARC_MARGIN = 0.30
    ARC_SCALE = 30.0
    CHECKPOINT_EVERY_EPOCHS = 5
    VAL_SPLIT = 0.15           # frazione di SOGGETTI (non campioni) per validazione
    SEED = 42

    VERIFICATION_THRESHOLD = 0.55   # da ricalibrare via EER sul proprio dataset

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        for d in [self.CHECKPOINT_DIR, self.FINAL_MODEL_DIR]:
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()


# ============================================================
# 2) FRIT - Finite Ridgelet Transform (handcrafted)
# ============================================================
# Radon transform (proiezioni su N angoli) + DWT 1-D (Haar) sulle
# proiezioni -> "ridgelet coefficients" -> inverse Radon per
# riportarli nel dominio spaziale come mappe allineate all'immagine.
# Approssimazione standard della FRIT esatta (che richiederebbe
# dimensioni prime), nessun parametro appreso: puramente handcrafted.
# Completata con gradienti Sobel/Laplaciano come canali extra.

def compute_frit_channels(img: np.ndarray, n_angles: int = 18,
                           wavelet: str = "haar", dwt_level: int = 2) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = gray.astype(np.float32) / 255.0
    h, w = gray.shape

    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    sinogram = radon(gray, theta=theta, circle=True)  # (n_offsets, n_angles)

    approx_bands, detail_bands = [], []
    for a in range(sinogram.shape[1]):
        proj = sinogram[:, a]
        coeffs = pywt.wavedec(proj, wavelet=wavelet, level=dwt_level)
        cA, cD = coeffs[0], coeffs[-1]
        up = lambda c: cv2.resize(c.reshape(-1, 1).astype(np.float32), (1, len(proj)),
                                   interpolation=cv2.INTER_LINEAR).flatten()
        approx_bands.append(up(cA))
        detail_bands.append(up(cD))

    sino_approx = np.stack(approx_bands, axis=1)
    sino_detail = np.stack(detail_bands, axis=1)

    img_approx = cv2.resize(iradon(sino_approx, theta=theta, circle=True, filter_name="ramp").astype(np.float32), (w, h))
    img_detail = cv2.resize(iradon(sino_detail, theta=theta, circle=True, filter_name="ramp").astype(np.float32), (w, h))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    def _norm(x):
        x = x - x.mean()
        return np.clip(x / (4 * (x.std() + 1e-6)), -1, 1).astype(np.float32)

    return np.stack([_norm(img_approx), _norm(img_detail), _norm(gray - gray.mean()),
                      _norm(gx), _norm(gy), _norm(lap)], axis=0)  # (6, H, W)


# ============================================================
# 3) MODELLI
# ============================================================
class TextureBranch(nn.Module):
    """ViT-S/14 DINOv2: transformer CONGELATO + MLP addestrabile sul CLS token."""

    def __init__(self, out_dim: int = cfg.TEXTURE_EMBED_DIM, freeze_backbone: bool = True):
        super().__init__()
        # dynamic_img_size=True: il checkpoint DINOv2 e' a 518x518, ma qui
        # alimentiamo 224x224 (coerente col preprocessing); i positional
        # embedding vengono interpolati automaticamente.
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True,
        )
        feat_dim = self.backbone.num_features
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, out_dim)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()   # niente drift delle running stats interne
        return self

    def forward(self, x):
        ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with ctx:
            feats = self.backbone(x)
        if self.freeze_backbone:
            feats = feats.detach()
        return self.mlp(feats)


class CentralROIBranch(nn.Module):
    """MobileNetV3-Large sulla ROI centrale del palmo."""

    def __init__(self, out_dim: int = cfg.ROI_EMBED_DIM, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        self.features = net.features
        self.avgpool = net.avgpool
        in_feat = net.classifier[0].in_features
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(in_feat, 512), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(512, out_dim)
        )

    def forward(self, x):
        f = self.avgpool(self.features(x)).flatten(1)
        return self.head(f)


class _KnuckleCNN(nn.Module):
    """CNN leggero condiviso, applicato a ciascuna nocca (6 canali FRIT in input)."""

    def __init__(self, in_channels: int = 6, feat_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, feat_dim)

    def forward(self, x):
        return self.proj(self.net(x).flatten(1))


class KnuckleFRITBranch(nn.Module):
    """N nocche (canali FRIT) -> CNN condiviso -> attention-pooling (gestisce nocche mancanti)."""

    def __init__(self, n_knuckles: int, out_dim: int = cfg.KNUCKLE_EMBED_DIM, in_channels: int = 6):
        super().__init__()
        self.n_knuckles = n_knuckles
        self.cnn = _KnuckleCNN(in_channels=in_channels, feat_dim=128)
        self.attn = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.out_proj = nn.Linear(128, out_dim)

    def forward(self, x, mask=None):
        b, n, c, h, w = x.shape
        feats = self.cnn(x.view(b * n, c, h, w)).view(b, n, -1)
        scores = self.attn(feats).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return self.out_proj((feats * weights).sum(dim=1))


class PalmEmbeddingNet(nn.Module):
    """Fusione dei 3 branch -> embedding biometrico L2-normalizzato."""

    def __init__(self, n_knuckles: int, embedding_dim: int = cfg.EMBEDDING_DIM,
                 freeze_vit: bool = True, freeze_mobilenet: bool = False):
        super().__init__()
        self.texture_branch = TextureBranch(cfg.TEXTURE_EMBED_DIM, freeze_vit)
        self.roi_branch = CentralROIBranch(cfg.ROI_EMBED_DIM, freeze_backbone=freeze_mobilenet)
        self.knuckle_branch = KnuckleFRITBranch(n_knuckles, cfg.KNUCKLE_EMBED_DIM)

        fusion_in = cfg.TEXTURE_EMBED_DIM + cfg.ROI_EMBED_DIM + cfg.KNUCKLE_EMBED_DIM
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, embedding_dim),
        )

    def forward(self, palm_hand, palm_roi, knuckles, knuckle_mask=None):
        t = self.texture_branch(palm_hand)
        r = self.roi_branch(palm_roi)
        k = self.knuckle_branch(knuckles, mask=knuckle_mask)
        emb = self.fusion(torch.cat([t, r, k], dim=1))
        return F.normalize(emb, p=2, dim=1)


class ArcMarginHead(nn.Module):
    """Testa ArcFace (solo training): margine angolare additivo per un embedding discriminativo."""

    def __init__(self, embedding_dim: int, n_classes: int, scale: float = cfg.ARC_SCALE, margin: float = cfg.ARC_MARGIN):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, embedding_dim) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, embeddings, labels):
        w = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(embeddings, w).clamp(-1 + 1e-7, 1 - 1e-7)
        target_logit = torch.cos(torch.acos(cos_theta) + self.margin)
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1.0)
        return (one_hot * target_logit + (1.0 - one_hot) * cos_theta) * self.scale


# ============================================================
# 4) DATASET
# ============================================================
def rgb_transform(train: bool) -> T.Compose:
    steps = [T.ToPILImage()]
    if train:
        # NB: nessun flip qui, preProcessing.py canonicalizza gia' L/R.
        steps += [T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                  T.RandomApply([T.GaussianBlur(3)], p=0.1)]
    steps += [T.ToTensor(), T.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)]
    return T.Compose(steps)


class PalmBiometricDataset(Dataset):
    """Legge direttamente la struttura di output di preProcessing.py (solo campioni palmari)."""

    def __init__(self, data_dir, subject_ids=None, train: bool = True, frit_cache: bool = True):
        self.data_dir = Path(data_dir)
        self.frit_cache = frit_cache
        self._frit_mem_cache = {}
        self.rgb_tf = rgb_transform(train)

        all_subject_dirs = sorted(d for d in self.data_dir.iterdir() if d.is_dir())
        if subject_ids is not None:
            allowed = set(subject_ids)
            all_subject_dirs = [d for d in all_subject_dirs if d.name in allowed]

        self.subject_to_label = {d.name: i for i, d in enumerate(all_subject_dirs)}
        self.samples = []

        for sdir in all_subject_dirs:
            for meta_path in sorted(sdir.glob("*_metadata.json")):
                meta = json.load(open(meta_path, "r", encoding="utf-8"))
                if meta.get("is_dorsal", False):
                    continue
                base = meta_path.name.replace("_metadata.json", "")
                hand_path, roi_path = sdir / f"{base}_palm_hand.png", sdir / f"{base}_palm_roi.png"
                if not (hand_path.exists() and roi_path.exists()):
                    continue
                knuckle_paths = [p for p in sorted(sdir.glob(f"{base}_palm_*.png"))
                                  if p.name not in (hand_path.name, roi_path.name)]
                if not knuckle_paths:
                    continue
                self.samples.append({"subject": sdir.name, "hand_path": hand_path,
                                      "roi_path": roi_path, "knuckle_paths": knuckle_paths})

        self.n_knuckles_max = max((len(s["knuckle_paths"]) for s in self.samples), default=12)

    def __len__(self):
        return len(self.samples)

    @property
    def num_classes(self):
        return len(self.subject_to_label)

    def _load_rgb(self, path):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return self.rgb_tf(img)

    def _load_knuckle_frit(self, path):
        key = str(path)
        if self.frit_cache and key in self._frit_mem_cache:
            return self._frit_mem_cache[key]
        channels = compute_frit_channels(cv2.imread(str(path)))
        if self.frit_cache:
            self._frit_mem_cache[key] = channels
        return channels

    def __getitem__(self, idx):
        s = self.samples[idx]
        knuckle_feats = np.zeros((self.n_knuckles_max, 6, *cfg.PALM_KNUCKLE_SIZE), dtype=np.float32)
        knuckle_mask = np.zeros((self.n_knuckles_max,), dtype=np.float32)
        for i, kp in enumerate(s["knuckle_paths"][: self.n_knuckles_max]):
            knuckle_feats[i] = self._load_knuckle_frit(kp)
            knuckle_mask[i] = 1.0

        return {
            "palm_hand": self._load_rgb(s["hand_path"]),
            "palm_roi": self._load_rgb(s["roi_path"]),
            "knuckles": torch.from_numpy(knuckle_feats),
            "knuckle_mask": torch.from_numpy(knuckle_mask),
            "label": torch.tensor(self.subject_to_label[s["subject"]], dtype=torch.long),
            "subject_id": s["subject"],
        }


def split_subjects(data_dir, val_split: float = cfg.VAL_SPLIT, seed: int = cfg.SEED):
    """Split PER SOGGETTO (non per campione): evita che la stessa identita' finisca in train e val."""
    rng = np.random.RandomState(seed)
    subjects = sorted(d.name for d in Path(data_dir).iterdir() if d.is_dir())
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * val_split))
    return subjects[n_val:], subjects[:n_val]   # train, val


# ============================================================
# 5) LOSS / METRICHE DI VERIFICA BIOMETRICA
# ============================================================
def triplet_loss(anchor, positive, negative, margin: float = 0.3):
    d_pos = 1 - F.cosine_similarity(anchor, positive)
    d_neg = 1 - F.cosine_similarity(anchor, negative)
    return F.relu(d_pos - d_neg + margin).mean()


@torch.no_grad()
def build_verification_pairs(labels, max_pairs: int = 20000, seed: int = 42):
    rng = np.random.RandomState(seed)
    by_subject = {}
    for i, lab in enumerate(labels):
        by_subject.setdefault(lab, []).append(i)

    pos_pairs = []
    for idxs in by_subject.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pos_pairs.append((idxs[a], idxs[b]))
    rng.shuffle(pos_pairs)
    pos_pairs = pos_pairs[: max_pairs // 2]

    subjects = list(by_subject.keys())
    if len(subjects) < 2:
        return pos_pairs, []   # impossibile formare coppie impostor

    neg_pairs, attempts = [], 0
    while len(neg_pairs) < len(pos_pairs) and attempts < max_pairs * 20:
        s1, s2 = rng.choice(subjects, 2, replace=False)
        neg_pairs.append((rng.choice(by_subject[s1]), rng.choice(by_subject[s2])))
        attempts += 1
    return pos_pairs, neg_pairs


@torch.no_grad()
def compute_eer(embeddings: torch.Tensor, subject_ids: list[str]):
    """EER su verifica a coppie, tipicamente su soggetti MAI visti in training (open-set)."""
    pos_pairs, neg_pairs = build_verification_pairs(subject_ids)
    if not pos_pairs or not neg_pairs:
        return None

    emb = F.normalize(embeddings, dim=1)
    sims = lambda pairs: (emb[[p[0] for p in pairs]] * emb[[p[1] for p in pairs]]).sum(1).cpu().numpy()
    genuine, impostor = sims(pos_pairs), sims(neg_pairs)

    thresholds = np.linspace(-1, 1, 500)
    fars = np.array([(impostor >= t).mean() for t in thresholds])
    frrs = np.array([(genuine < t).mean() for t in thresholds])
    idx = np.argmin(np.abs(fars - frrs))

    return {"eer": float((fars[idx] + frrs[idx]) / 2), "eer_threshold": float(thresholds[idx]),
            "n_genuine": len(genuine), "n_impostor": len(impostor)}