"""
============================================================
PALM_RUN - Training + Inferenza (CLI unica)
============================================================
Uso:
  # training (con checkpoint periodici + salvataggio best/final)
  python palm_run.py train --data_dir dataset_preprocessed --epochs 60

  # riprendere un training interrotto
  python palm_run.py train --resume checkpoints/palm_embedding_epoch020.pt

  # verifica 1:1 tra due acquisizioni gia' preprocessate
  python palm_run.py verify --checkpoint models_final/palm_embedding_best.pt \
      --base1 dataset_preprocessed/0001/0001_palmar_001 \
      --base2 dataset_preprocessed/0001/0001_palmar_002
============================================================
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from palm_core import (
    cfg, compute_frit_channels, rgb_transform,
    PalmEmbeddingNet, ArcMarginHead,
    PalmBiometricDataset, split_subjects, compute_eer,
)


# ============================================================
# TRAINING
# ============================================================
def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, epoch, model, head, optimizer, scheduler, best_eer, n_knuckles, subject_to_label):
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(), "head_state": head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_eer": best_eer, "n_knuckles": n_knuckles,
        "subject_to_label": subject_to_label, "embedding_dim": cfg.EMBEDDING_DIM,
    }, path)


def load_checkpoint(path, model, head, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    head.load_state_dict(ckpt["head_state"])
    if optimizer is not None and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


def train_one_epoch(model, head, loader, optimizer, criterion, device, epoch, log_every=20):
    model.train(); head.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        palm_hand = batch["palm_hand"].to(device, non_blocking=True)
        palm_roi = batch["palm_roi"].to(device, non_blocking=True)
        knuckles = batch["knuckles"].to(device, non_blocking=True)
        knuckle_mask = batch["knuckle_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        emb = model(palm_hand, palm_roi, knuckles, knuckle_mask)
        logits = head(emb, labels)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] + list(head.parameters()), 5.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += labels.size(0)
        if step % log_every == 0:
            print(f"  [epoch {epoch}] step {step}/{len(loader)} loss={loss.item():.4f} "
                  f"acc={total_correct/max(1,total_n):.4f}")

    return total_loss / max(1, total_n), total_correct / max(1, total_n), time.time() - t0


@torch.no_grad()
def evaluate_open_set(model, loader, device):
    model.eval()
    all_emb, all_subj = [], []
    for batch in loader:
        emb = model(batch["palm_hand"].to(device), batch["palm_roi"].to(device),
                    batch["knuckles"].to(device), batch["knuckle_mask"].to(device))
        all_emb.append(emb.cpu())
        all_subj.extend(batch["subject_id"])
    if not all_emb:
        return None
    return compute_eer(torch.cat(all_emb, dim=0), all_subj)


def cmd_train(args):
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Device: {device}")

    train_subjects, val_subjects = split_subjects(args.data_dir)
    print(f"Soggetti train: {len(train_subjects)} | Soggetti validation (open-set): {len(val_subjects)}")

    train_ds = PalmBiometricDataset(args.data_dir, subject_ids=train_subjects, train=True)
    val_ds = PalmBiometricDataset(args.data_dir, subject_ids=val_subjects, train=False)
    val_ds.n_knuckles_max = train_ds.n_knuckles_max

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=cfg.NUM_WORKERS, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print(f"Campioni train: {len(train_ds)} | Campioni validation: {len(val_ds)} | "
          f"Nocche per campione: {train_ds.n_knuckles_max}")

    model = PalmEmbeddingNet(train_ds.n_knuckles_max, cfg.EMBEDDING_DIM,
                              freeze_vit=args.freeze_vit, freeze_mobilenet=False).to(device)
    head = ArcMarginHead(cfg.EMBEDDING_DIM, train_ds.num_classes).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch, best_eer = 1, 1.0
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model, head, optimizer, scheduler)
        start_epoch = ckpt["epoch"] + 1
        best_eer = ckpt.get("best_eer", 1.0)
        print(f"Ripreso da {args.resume} (epoch {ckpt['epoch']}, best_eer={best_eer:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc, dt = train_one_epoch(model, head, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} - loss={train_loss:.4f} acc={train_acc:.4f} "
              f"({dt:.1f}s) lr={scheduler.get_last_lr()[0]:.2e}")

        val_metrics = evaluate_open_set(model, val_loader, device)
        if val_metrics:
            print(f"  [val open-set] EER={val_metrics['eer']*100:.2f}% @thr={val_metrics['eer_threshold']:.3f} "
                  f"(pairs: {val_metrics['n_genuine']} genuine / {val_metrics['n_impostor']} impostor)")
            if val_metrics["eer"] < best_eer:
                best_eer = val_metrics["eer"]
                best_path = cfg.FINAL_MODEL_DIR / "palm_embedding_best.pt"
                save_checkpoint(best_path, epoch, model, head, optimizer, scheduler,
                                 best_eer, train_ds.n_knuckles_max, train_ds.subject_to_label)
                print(f"  -> Nuovo best model salvato ({best_path}) EER={best_eer*100:.2f}%")

        if epoch % cfg.CHECKPOINT_EVERY_EPOCHS == 0 or epoch == args.epochs:
            ckpt_path = cfg.CHECKPOINT_DIR / f"palm_embedding_epoch{epoch:03d}.pt"
            save_checkpoint(ckpt_path, epoch, model, head, optimizer, scheduler,
                             best_eer, train_ds.n_knuckles_max, train_ds.subject_to_label)
            print(f"  -> Checkpoint salvato: {ckpt_path}")

    final_path = cfg.FINAL_MODEL_DIR / "palm_embedding_final.pt"
    save_checkpoint(final_path, args.epochs, model, head, optimizer, scheduler,
                     best_eer, train_ds.n_knuckles_max, train_ds.subject_to_label)
    print(f"\nTraining completato. Modello finale: {final_path}")
    print(f"Miglior EER su validation open-set: {best_eer*100:.2f}%")


# ============================================================
# INFERENZA / VERIFICA
# ============================================================
class PalmVerifier:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or cfg.DEVICE
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.n_knuckles = ckpt["n_knuckles"]
        self.model = PalmEmbeddingNet(self.n_knuckles, ckpt.get("embedding_dim", cfg.EMBEDDING_DIM),
                                       freeze_vit=True).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.rgb_tf = rgb_transform(train=False)
        print(f"Modello caricato da {checkpoint_path} (epoch {ckpt['epoch']}, "
              f"best_eer={ckpt.get('best_eer', float('nan')):.4f})")

    def _load_sample(self, base_path):
        base_path = Path(base_path)
        folder, prefix = base_path.parent, base_path.name
        hand_path, roi_path = folder / f"{prefix}_palm_hand.png", folder / f"{prefix}_palm_roi.png"
        knuckle_paths = [p for p in sorted(folder.glob(f"{prefix}_palm_*.png"))
                          if p.name not in (hand_path.name, roi_path.name)]
        if not hand_path.exists() or not roi_path.exists():
            raise FileNotFoundError(f"File mancanti per {base_path} (palm_hand/palm_roi)")

        palm_hand = self.rgb_tf(cv2.cvtColor(cv2.imread(str(hand_path)), cv2.COLOR_BGR2RGB)).unsqueeze(0)
        palm_roi = self.rgb_tf(cv2.cvtColor(cv2.imread(str(roi_path)), cv2.COLOR_BGR2RGB)).unsqueeze(0)

        knuckles = torch.zeros(1, self.n_knuckles, 6, *cfg.PALM_KNUCKLE_SIZE)
        mask = torch.zeros(1, self.n_knuckles)
        for i, kp in enumerate(knuckle_paths[: self.n_knuckles]):
            knuckles[0, i] = torch.from_numpy(compute_frit_channels(cv2.imread(str(kp))))
            mask[0, i] = 1.0

        return (palm_hand.to(self.device), palm_roi.to(self.device),
                knuckles.to(self.device), mask.to(self.device))

    @torch.no_grad()
    def embed(self, base_path) -> torch.Tensor:
        palm_hand, palm_roi, knuckles, mask = self._load_sample(base_path)
        return self.model(palm_hand, palm_roi, knuckles, mask).squeeze(0).cpu()

    @torch.no_grad()
    def verify(self, base_path1, base_path2, threshold: float = cfg.VERIFICATION_THRESHOLD):
        emb1, emb2 = self.embed(base_path1), self.embed(base_path2)
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return {"same_subject": bool(similarity >= threshold), "similarity": similarity, "threshold": threshold}

    @torch.no_grad()
    def identify(self, probe_base_path, gallery: dict[str, torch.Tensor], threshold: float = cfg.VERIFICATION_THRESHOLD):
        """gallery: {subject_id: embedding} pre-calcolata. Ritorna il piu' simile, o None se sotto soglia."""
        probe_emb = self.embed(probe_base_path)
        best_subject, best_sim = None, -1.0
        for subject_id, gallery_emb in gallery.items():
            sim = F.cosine_similarity(probe_emb.unsqueeze(0), gallery_emb.unsqueeze(0)).item()
            if sim > best_sim:
                best_subject, best_sim = subject_id, sim
        if best_sim < threshold:
            return {"identified_subject": None, "similarity": best_sim}
        return {"identified_subject": best_subject, "similarity": best_sim}


def cmd_verify(args):
    verifier = PalmVerifier(args.checkpoint)
    result = verifier.verify(args.base1, args.base2, threshold=args.threshold)
    print(json.dumps(result, indent=2))
    verdict = "STESSO SOGGETTO" if result["same_subject"] else "SOGGETTI DIVERSI"
    print(f"\n>>> {verdict}  (similarity={result['similarity']:.4f}, soglia={result['threshold']:.4f})")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Training + inferenza embedding biometrico palmo")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="addestra il modello di embedding")
    p_train.add_argument("--data_dir", default=str(cfg.DATA_DIR))
    p_train.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p_train.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p_train.add_argument("--lr", type=float, default=cfg.LR)
    p_train.add_argument("--freeze_vit", type=lambda x: x.lower() != "false", default=True)
    p_train.add_argument("--resume", default=None)
    p_train.set_defaults(func=cmd_train)

    p_verify = sub.add_parser("verify", help="verifica 1:1 tra due acquisizioni preprocessate")
    p_verify.add_argument("--checkpoint", required=True)
    p_verify.add_argument("--base1", required=True)
    p_verify.add_argument("--base2", required=True)
    p_verify.add_argument("--threshold", type=float, default=cfg.VERIFICATION_THRESHOLD)
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
