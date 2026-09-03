"""
============================================================
DORSAL_CORE - Config + Vein Enhancement + Modelli + Dataset + Loss/Metriche
============================================================
Pipeline di embedding biometrico per lo stream DORSO, coerente
con l'output di preProcessing.py:

  <output_dir>/<subject_id>/<subject>_<side>_<seq>_dorsal_hand.png      -> Swin-Tiny (frozen) + MLP
  <output_dir>/<subject_id>/<subject>_<side>_<seq>_dorsal_<knuckle>.png -> MobileNetV3-Large (ROI nocche)
  <output_dir>/<subject_id>/<subject>_<side>_<seq>_metadata.json        -> is_dorsal = True

Contenuto:
  1. CONFIG
  2. VEIN ENHANCEMENT (CLAHE forte + Frangi vesselness, handcrafted, no pesi appresi)
  3. MODELLI: DorsalTextureBranch (Swin-Tiny) / KnuckleMobileNetBranch (MobileNetV3-Large,
     attention-pooling multi-nocca) + DorsalEmbeddingNet (fusione) + ArcMarginHead (solo training)
  4. DATASET: legge direttamente l'output di preProcessing.py (solo campioni dorsali)
  5. LOSS / METRICHE: ArcFace+CE, triplet opzionale, EER open-set
============================================================
"""

from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from skimage.filters import frangi
import timm


# ============================================================
# 1) CONFIG
# ============================================================
class Config:
    DATA_DIR = Path("./dataset_preprocessed")     # output di preProcessing.py
    CHECKPOINT_DIR = Path("./checkpoints_dorsal")
    FINAL_MODEL_DIR = Path("./models_final_dorsal")

    SWIN_SIZE = (224, 224)
    DORSAL_KNUCKLE_SIZE = (224, 224)

    EMBEDDING_DIM = 256
    TEXTURE_EMBED_DIM = 128     # Swin-Tiny (forma + vene)
    KNUCKLE_EMBED_DIM = 128     # MobileNetV3-Large (ROI nocche)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 60
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    ARC_MARGIN = 0.30
    ARC_SCALE = 30.0
    CHECKPOINT_EVERY_EPOCHS = 5
    VAL_SPLIT = 0.15           # usato solo dal training semplice
    SEED = 42

    # Nested K-Fold (sempre PER SOGGETTO, mai per singolo campione)
    OUTER_FOLDS = 5
    INNER_FOLDS = 4
    INNER_EPOCHS = 25
    OUTER_EPOCHS = 60

    VERIFICATION_THRESHOLD = 0.55   # da ricalibrare via EER sul proprio dataset

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        for d in [self.CHECKPOINT_DIR, self.FINAL_MODEL_DIR]:
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()


# ============================================================
# 2) VEIN ENHANCEMENT (handcrafted, no pesi appresi)
# ============================================================
# preProcessing.py applica gia' una CLAHE mite (illuminazione) su
# palm_hand/dorsal_hand in modo identico per i due stream, per
# rendere il preprocessing agnostico rispetto al backbone a valle.
# Qui, SOLO per lo stream dorso (branch forma+vene), aggiungiamo
# un potenziamento mirato a far risaltare i vasi sanguigni
# sottocutanei visibili sul dorso della mano:
#   1) CLAHE aggressiva sul canale L (Lab) per aumentare il
#      contrasto locale della pelle;
#   2) filtro di Frangi (vesselness/ridge detector multiscala,
#      handcrafted, lo stesso principio usato in angiografia)
#      applicato al canale verde (il piu' sensibile al contrasto
#      vena/pelle in immagini RGB);
#   3) blending della mappa di vesselness con l'immagine originale
#      per esaltare le vene senza distruggere la forma della mano
#      (fondamentale perche' lo stesso branch impara anche la forma).
def enhance_veins(img_rgb: np.ndarray, clahe_clip: float = 3.0,
                   vessel_scales=(1, 2, 3, 4), vessel_weight: float = 0.55) -> np.ndarray:
    """img_rgb: uint8 HxWx3 RGB. Ritorna uint8 HxWx3 RGB con vene esaltate."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)
    lab_eq = cv2.merge([L_eq, a, b])
    contrast_rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # Canale verde: massimo assorbimento dell'emoglobina -> massimo
    # contrasto vena/pelle nello spettro visibile RGB.
    green = contrast_rgb[:, :, 1].astype(np.float64) / 255.0

    # Le vene sono strutture scure e sottili rispetto alla pelle
    # circostante -> invertiamo il canale cosi' che Frangi (pensato
    # per strutture chiare tipo vasi in angiografia) le rilevi come ridge.
    vesselness = frangi(1.0 - green, sigmas=vessel_scales, black_ridges=False)
    if vesselness.max() > 1e-8:
        vesselness = vesselness / vesselness.max()
    vesselness_u8 = (vesselness * 255).astype(np.uint8)
    vesselness_rgb = cv2.cvtColor(vesselness_u8, cv2.COLOR_GRAY2RGB)

    enhanced = cv2.addWeighted(
        contrast_rgb, 1.0 - vessel_weight * 0.5,
        vesselness_rgb, vessel_weight * 0.5, 0,
    )
    return np.clip(enhanced, 0, 255).astype(np.uint8)


# ============================================================
# 3) MODELLI
# ============================================================
class DorsalTextureBranch(nn.Module):
    """Swin-Tiny: transformer CONGELATO + MLP addestrabile, su input con vene esaltate."""

    def __init__(self, out_dim: int = cfg.TEXTURE_EMBED_DIM, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0,
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
        # Se x ha gia' 2 dimensioni (B, feat_dim) e' una embedding Swin
        # PRE-CALCOLATA (vedi precompute_swin_embeddings.py): saltiamo
        # backbone + enhance_veins (Frangi), che su CPU sono la parte
        # piu' costosa di questo branch.
        if x.dim() == 2:
            feats = x
        else:
            ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
            with ctx:
                feats = self.backbone(x)
            if self.freeze_backbone:
                feats = feats.detach()
        return self.mlp(feats)


class _KnuckleMobileNet(nn.Module):
    """MobileNetV3-Large condiviso, applicato a ciascun ritaglio di nocca (RGB)."""

    def __init__(self, feat_dim: int = 128, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        self.features = net.features
        self.avgpool = net.avgpool
        in_feat = net.classifier[0].in_features
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(in_feat, 256), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(256, feat_dim)
        )

    def forward(self, x):
        f = self.avgpool(self.features(x)).flatten(1)
        return self.proj(f)


class KnuckleMobileNetBranch(nn.Module):
    """N nocche dorsali (crop MediaPipe, RGB) -> MobileNetV3-Large condiviso -> attention-pooling."""

    def __init__(self, n_knuckles: int, out_dim: int = cfg.KNUCKLE_EMBED_DIM,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.n_knuckles = n_knuckles
        self.cnn = _KnuckleMobileNet(feat_dim=128, pretrained=pretrained, freeze_backbone=freeze_backbone)
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


class DorsalEmbeddingNet(nn.Module):
    """Fusione dei 2 branch (forma+vene / nocche) -> embedding biometrico L2-normalizzato."""

    def __init__(self, n_knuckles: int, embedding_dim: int = cfg.EMBEDDING_DIM,
                 freeze_swin: bool = True, freeze_mobilenet: bool = False):
        super().__init__()
        self.texture_branch = DorsalTextureBranch(cfg.TEXTURE_EMBED_DIM, freeze_swin)
        self.knuckle_branch = KnuckleMobileNetBranch(n_knuckles, cfg.KNUCKLE_EMBED_DIM,
                                                       freeze_backbone=freeze_mobilenet)

        fusion_in = cfg.TEXTURE_EMBED_DIM + cfg.KNUCKLE_EMBED_DIM
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, embedding_dim),
        )

    def forward(self, dorsal_hand, knuckles, knuckle_mask=None):
        t = self.texture_branch(dorsal_hand)
        k = self.knuckle_branch(knuckles, mask=knuckle_mask)
        emb = self.fusion(torch.cat([t, k], dim=1))
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
class _EnhanceVeinsTransform:
    """Wrapper picklabile per enhance_veins, necessario perche' su Windows
    il DataLoader con num_workers>0 usa multiprocessing 'spawn', che richiede
    che tutti gli oggetti passati ai worker (incluse le trasformazioni) siano
    pickle-abili. Una lambda o una funzione locale/nested non lo sono."""

    def __call__(self, img):
        return enhance_veins(img)


def dorsal_hand_transform(train: bool) -> T.Compose:
    """Trasformazione per l'immagine mano-intera: applica il potenziamento vene
    PRIMA della normalizzazione ImageNet, cosi' il branch Swin vede sia la
    forma sia i vasi sanguigni esaltati."""
    steps = [_EnhanceVeinsTransform(), T.ToPILImage()]
    if train:
        # NB: nessun flip qui, preProcessing.py canonicalizza gia' L/R.
        steps += [T.ColorJitter(brightness=0.1, contrast=0.1),
                  T.RandomApply([T.GaussianBlur(3)], p=0.1)]
    steps += [T.ToTensor(), T.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)]
    return T.Compose(steps)


def knuckle_transform(train: bool) -> T.Compose:
    """Trasformazione per i ritagli di nocca (input diretto a MobileNetV3, RGB standard)."""
    steps = [T.ToPILImage()]
    if train:
        steps += [T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                  T.RandomApply([T.GaussianBlur(3)], p=0.1)]
    steps += [T.ToTensor(), T.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)]
    return T.Compose(steps)


class DorsalBiometricDataset(Dataset):
    """Legge direttamente la struttura di output di preProcessing.py (solo campioni dorsali)."""

    def __init__(self, data_dir, subject_ids=None, train: bool = True,
                 swin_embed_cache: dict | None = None):
        self.data_dir = Path(data_dir)
        self.hand_tf = dorsal_hand_transform(train)
        self.knuckle_tf = knuckle_transform(train)
        # Dict {percorso_assoluto_dorsal_hand: np.ndarray(feat_dim,)} prodotto
        # da precompute_swin_embeddings.py. Se presente, saltiamo sia
        # enhance_veins() (Frangi) sia il forward dello Swin (il backbone
        # e' sempre congelato) e restituiamo direttamente l'embedding.
        self.swin_embed_cache = swin_embed_cache

        all_subject_dirs = sorted(d for d in self.data_dir.iterdir() if d.is_dir())
        if subject_ids is not None:
            allowed = set(subject_ids)
            all_subject_dirs = [d for d in all_subject_dirs if d.name in allowed]

        self.subject_to_label = {d.name: i for i, d in enumerate(all_subject_dirs)}
        self.samples = []

        for sdir in all_subject_dirs:
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
                self.samples.append({"subject": sdir.name, "hand_path": hand_path,
                                      "knuckle_paths": knuckle_paths})

        self.n_knuckles_max = max((len(s["knuckle_paths"]) for s in self.samples), default=12)

    def __len__(self):
        return len(self.samples)

    @property
    def num_classes(self):
        return len(self.subject_to_label)

    def _load_rgb(self, path, tf):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return tf(img)

    def __getitem__(self, idx):
        s = self.samples[idx]
        c, h, w = 3, *cfg.DORSAL_KNUCKLE_SIZE
        knuckle_feats = torch.zeros((self.n_knuckles_max, c, h, w), dtype=torch.float32)
        knuckle_mask = np.zeros((self.n_knuckles_max,), dtype=np.float32)
        for i, kp in enumerate(s["knuckle_paths"][: self.n_knuckles_max]):
            knuckle_feats[i] = self._load_rgb(kp, self.knuckle_tf)
            knuckle_mask[i] = 1.0

        if self.swin_embed_cache is not None:
            key = str(s["hand_path"].resolve())
            if key not in self.swin_embed_cache:
                raise KeyError(
                    f"Nessuna embedding Swin precalcolata per {key}. "
                    f"Rilancia precompute_swin_embeddings.py sull'intero data_dir."
                )
            dorsal_hand_val = torch.from_numpy(self.swin_embed_cache[key]).float()
        else:
            dorsal_hand_val = self._load_rgb(s["hand_path"], self.hand_tf)

        return {
            "dorsal_hand": dorsal_hand_val,
            "knuckles": knuckle_feats,
            "knuckle_mask": torch.from_numpy(knuckle_mask),
            "label": torch.tensor(self.subject_to_label[s["subject"]], dtype=torch.long),
            "subject_id": s["subject"],
        }


def list_subjects(data_dir):
    """Ritorna gli ID soggetto ordinati presenti nel dataset."""
    return sorted(d.name for d in Path(data_dir).iterdir() if d.is_dir())


def kfold_subject_splits(subject_ids, n_splits: int, seed: int = cfg.SEED):
    """
    K-Fold deterministico PER SOGGETTO.

    Ogni split restituisce:
        (fold_index, train_subjects, test_subjects)

    Non viene mai spezzata l'identita' tra train e test, evitando leakage
    biometrico tra acquisizioni dello stesso soggetto.
    """
    subjects = list(subject_ids)
    if n_splits < 2:
        raise ValueError("n_splits deve essere >= 2")
    if len(subjects) < n_splits:
        raise ValueError(
            f"Servono almeno {n_splits} soggetti per il K-Fold, trovati {len(subjects)}"
        )

    rng = np.random.RandomState(seed)
    subjects = np.array(sorted(subjects), dtype=object)
    rng.shuffle(subjects)

    fold_sizes = np.full(n_splits, len(subjects) // n_splits, dtype=int)
    fold_sizes[: len(subjects) % n_splits] += 1

    current = 0
    for fold_idx, fold_size in enumerate(fold_sizes, start=1):
        test_subjects = subjects[current: current + fold_size].tolist()
        train_subjects = np.concatenate(
            [subjects[:current], subjects[current + fold_size:]]
        ).tolist()
        current += fold_size
        yield fold_idx, train_subjects, test_subjects


def split_subjects(data_dir, val_split: float = cfg.VAL_SPLIT, seed: int = cfg.SEED):
    """
    Split semplice PER SOGGETTO.

    Mantiene la compatibilita' con il comando train classico.
    Per una valutazione scientificamente corretta usare nested_cv.
    """
    rng = np.random.RandomState(seed)
    subjects = list_subjects(data_dir)
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