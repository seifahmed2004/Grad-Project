#!/usr/bin/env python3
"""
Age Prediction – EfficientNet-B4 Multi-Task (Regression + Classification)
HPC-ready: robust error handling, reproducibility, checkpointing, clean logging.
Dataset layout expected:  age-prediction/{train,test}/<age_int>/<image_file>
"""

import os
import sys
import time
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
import cv2
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – required on HPC (no display)
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 0.  CLI Arguments
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Age Prediction Training")
    p.add_argument("--data",        type=str, default="age-prediction",
                   help="Root folder of the dataset (default: age-prediction)")
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--batch_size",  type=int, default=48)
    p.add_argument("--num_workers", type=int, default=2)   # matched to cpus-per-task=2
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output_dir",  type=str, default="outputs")
    p.add_argument("--resume",      type=str, default=None,
                   help="Path to checkpoint .pth to resume from")
    return p.parse_args()

# ──────────────────────────────────────────────
# 1.  Reproducibility
# ──────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ──────────────────────────────────────────────
# 2.  Logging
# ──────────────────────────────────────────────
def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("age_train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

# ──────────────────────────────────────────────
# 3.  Face crop (safe, non-crashing)
# ──────────────────────────────────────────────
_face_cascade = None

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(xml)
    return _face_cascade

def crop_face_pil(image: Image.Image) -> Image.Image:
    try:
        img_np  = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces   = get_face_cascade().detectMultiScale(
                      gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return image
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.25 * max(w, h))
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(img_np.shape[1], x + w + pad)
        y2  = min(img_np.shape[0], y + h + pad)
        return Image.fromarray(img_np[y1:y2, x1:x2])
    except Exception:
        return image   # never crash a DataLoader worker

# ──────────────────────────────────────────────
# 4.  Age → bin mapping
# ──────────────────────────────────────────────
BIN_EDGES = [12, 19, 29, 39, 49, 59]   # produces 7 bins

def age_to_bin(age: float) -> int:
    for i, edge in enumerate(BIN_EDGES):
        if age <= edge:
            return i
    return len(BIN_EDGES)

# ──────────────────────────────────────────────
# 5.  Dataset
# ──────────────────────────────────────────────
class AgeDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train",
                 transform=None, crop_face: bool = False):
        self.transform  = transform
        self.crop_face  = crop_face
        self.samples    = []
        valid_ext       = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        root_dir = str(root_dir)
        for root, _, files in os.walk(root_dir):
            root_fixed = root.replace("\\", "/")
            if f"/{split}/" not in root_fixed:
                continue
            folder_name = os.path.basename(root)
            if not folder_name.isdigit():
                continue
            age = float(folder_name)
            for f in files:
                if f.lower().endswith(valid_ext):
                    self.samples.append((os.path.join(root, f), age))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found for split='{split}' under '{root_dir}'.\n"
                f"Expected layout: {root_dir}/{split}/<age_int>/<image>")

        ages = [a for _, a in self.samples]
        self.min_age = min(ages)
        self.max_age = max(ages)

        age_counts: dict = {}
        for a in ages:
            age_counts[int(a)] = age_counts.get(int(a), 0) + 1
        max_count = max(age_counts.values())
        self.sample_weights = [max_count / age_counts[int(a)] for a in ages]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]
        for _ in range(5):
            try:
                image = Image.open(img_path).convert("RGB")
                break
            except (UnidentifiedImageError, OSError):
                idx = (idx + 1) % len(self.samples)
                img_path, age = self.samples[idx]
        else:
            image = Image.new("RGB", (224, 224))

        if self.crop_face:
            image = crop_face_pil(image)
        if self.transform:
            image = self.transform(image)

        return (image,
                torch.tensor(age, dtype=torch.float32),
                torch.tensor(age_to_bin(age), dtype=torch.long))

# ──────────────────────────────────────────────
# 6.  Model
# ──────────────────────────────────────────────
class AgeEfficientNetB4(nn.Module):
    def __init__(self, min_age: float, max_age: float, num_bins: int = 7):
        super().__init__()
        self.min_age  = min_age
        self.max_age  = max_age
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=True,
            num_classes=0, global_pool="avg")
        in_features = self.backbone.num_features   # 1792

        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.age_head = nn.Linear(256, 1)
        self.bin_head = nn.Linear(256, num_bins)

    def forward(self, x, clamp_output: bool = False):
        x       = self.backbone(x)
        x       = self.shared(x)
        age_out = self.age_head(x).squeeze(1)
        if clamp_output:
            age_out = age_out.clamp(self.min_age, self.max_age)
        return age_out, self.bin_head(x)

# ──────────────────────────────────────────────
# 7.  Progressive unfreezing
# ──────────────────────────────────────────────
def unfreeze_stage2(model, logger):
    for block in list(model.backbone.blocks)[-2:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.backbone.conv_head.parameters():
        p.requires_grad = True
    for p in model.backbone.bn2.parameters():
        p.requires_grad = True
    logger.info("Stage 2: unfroze last 2 backbone blocks + conv_head + bn2.")

def unfreeze_stage3(model, logger):
    for p in model.backbone.parameters():
        p.requires_grad = True
    logger.info("Stage 3: entire backbone unfrozen.")

def build_optimizer_stage1(model):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4)

def build_optimizer_stage2(model):
    return torch.optim.AdamW([
        {"params": filter(lambda p: p.requires_grad,
                          model.backbone.parameters()), "lr": 3e-5},
        {"params": model.shared.parameters(),           "lr": 3e-4},
        {"params": model.age_head.parameters(),         "lr": 3e-4},
        {"params": model.bin_head.parameters(),         "lr": 3e-4},
    ], weight_decay=1e-4)

def build_optimizer_stage3(model):
    return torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.shared.parameters(),   "lr": 1e-4},
        {"params": model.age_head.parameters(), "lr": 1e-4},
        {"params": model.bin_head.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

# ──────────────────────────────────────────────
# 8.  Checkpoint helpers
# ──────────────────────────────────────────────
def save_checkpoint(state: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)   # atomic rename – safe if job is killed mid-write

def load_checkpoint(path: str, model, optimizer, scheduler, logger):
    ckpt           = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch    = ckpt["epoch"] + 1
    best_val_mae   = ckpt["best_val_mae"]
    train_mae_list = ckpt.get("train_mae_list", [])
    val_mae_list   = ckpt.get("val_mae_list",   [])
    within3_list   = ckpt.get("within3_list",   [])
    logger.info(f"Resumed from '{path}' — starting at epoch {start_epoch}, "
                f"best_val_mae={best_val_mae:.4f}")
    return start_epoch, best_val_mae, train_mae_list, val_mae_list, within3_list

# ──────────────────────────────────────────────
# 9.  Plots
# ──────────────────────────────────────────────
def save_plots(train_mae_list, val_mae_list, within3_list, output_dir: Path):
    epochs_range = range(1, len(train_mae_list) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, train_mae_list, marker="o", label="Train MAE")
    ax.plot(epochs_range, val_mae_list,   marker="o", label="Val MAE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE")
    ax.set_title("Train vs Val MAE"); ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "train_vs_val_mae.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, within3_list, marker="o", label="Within ±3 Years")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy %")
    ax.set_title("Validation Accuracy Within ±3 Years")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "within_3_years_accuracy.png", dpi=150)
    plt.close(fig)

# ──────────────────────────────────────────────
# 10. Main
# ──────────────────────────────────────────────
def main():
    args   = parse_args()
    outdir = Path(args.output_dir)
    logger = setup_logger(outdir)
    set_seed(args.seed)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU : {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Validate dataset path ──
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Dataset path '{data_path}' does not exist. Aborting.")
        sys.exit(1)

    # ── Transforms ──
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Datasets ──
    logger.info("Loading datasets …")
    train_dataset = AgeDataset(data_path, split="train",
                               transform=train_tf, crop_face=True)
    val_dataset   = AgeDataset(data_path, split="test",
                               transform=val_tf,   crop_face=True)
    logger.info(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | "
                f"Age range: {train_dataset.min_age:.0f}–{train_dataset.max_age:.0f}")

    # ── Dataloaders ──
    # num_workers=2 matches cpus-per-task=2 in the SLURM script.
    # persistent_workers keeps workers alive between epochs (faster, less memory churn).
    nw = args.num_workers
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(train_dataset.sample_weights, dtype=torch.float32),
        num_samples = len(train_dataset),
        replacement = True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=nw,
                              pin_memory=True, drop_last=True,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=nw,
                              pin_memory=True,
                              persistent_workers=(nw > 0))

    # ── Class weights ──
    bin_counts = torch.zeros(7)
    for _, age in train_dataset.samples:
        bin_counts[age_to_bin(age)] += 1
    class_weights = bin_counts.sum() / (bin_counts + 1e-6)
    class_weights = (class_weights / class_weights.sum() * 7).to(device)
    logger.info(f"Bin counts   : {bin_counts.long().tolist()}")
    logger.info(f"Class weights: {[f'{w:.3f}' for w in class_weights.cpu().tolist()]}")

    # ── Model ──
    model = AgeEfficientNetB4(train_dataset.min_age,
                              train_dataset.max_age).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False    # Stage 1: train heads only

    reg_criterion = nn.HuberLoss(delta=5.0, reduction="mean")
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = build_optimizer_stage1(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    # ── Resume ──
    start_epoch    = 0
    best_val_mae   = float("inf")
    train_mae_list = []
    val_mae_list   = []
    within3_list   = []

    if args.resume:
        if not Path(args.resume).exists():
            logger.error(f"Checkpoint '{args.resume}' not found. Aborting.")
            sys.exit(1)
        start_epoch, best_val_mae, train_mae_list, val_mae_list, within3_list = \
            load_checkpoint(args.resume, model, optimizer, scheduler, logger)

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Progressive unfreezing
        if epoch == 3:
            unfreeze_stage2(model, logger)
            optimizer = build_optimizer_stage2(model)
        elif epoch == 7:
            unfreeze_stage3(model, logger)
            optimizer = build_optimizer_stage3(model)

        # ── Train ──
        model.train()
        train_mae   = 0.0
        train_total = 0

        for batch_idx, (images, ages, age_bins) in enumerate(train_loader):
            images   = images.to(device,   non_blocking=True)
            ages     = ages.to(device,     non_blocking=True)
            age_bins = age_bins.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_ages, pred_bins = model(images)

            loss = reg_criterion(pred_ages, ages) + \
                   0.3 * cls_criterion(pred_bins, age_bins)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_mae   += torch.abs(pred_ages - ages).sum().item()
            train_total += ages.size(0)

            if batch_idx % 50 == 0:
                logger.info(f"  Epoch {epoch+1}/{args.epochs}  "
                            f"Batch {batch_idx}/{len(train_loader)}  "
                            f"Loss {loss.item():.4f}")

        train_mae /= train_total

        # ── Validate ──
        model.eval()
        val_mae  = 0.0
        within_3 = within_5 = total = 0

        with torch.no_grad():
            for images, ages, _ in val_loader:
                images = images.to(device, non_blocking=True)
                ages   = ages.to(device,   non_blocking=True)
                pred_ages, _ = model(images, clamp_output=True)
                diff     = torch.abs(pred_ages - ages)
                val_mae  += diff.sum().item()
                within_3 += (diff <= 3).sum().item()
                within_5 += (diff <= 5).sum().item()
                total    += ages.size(0)

        val_mae /= total
        acc3 = 100.0 * within_3 / total
        acc5 = 100.0 * within_5 / total
        scheduler.step(val_mae)

        train_mae_list.append(train_mae)
        val_mae_list.append(val_mae)
        within3_list.append(acc3)

        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"Train MAE {train_mae:.3f} | Val MAE {val_mae:.3f} | "
            f"±3yr {acc3:.1f}% | ±5yr {acc5:.1f}% | {elapsed:.0f}s")

        # ── Save best model ──
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(),
                       outdir / "best_age_efficientnet_b4.pth")
            logger.info(f"  ★  New best saved  (val_mae={best_val_mae:.4f})")

        # ── Save resumable checkpoint every epoch ──
        save_checkpoint({
            "epoch":          epoch,
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "best_val_mae":   best_val_mae,
            "train_mae_list": train_mae_list,
            "val_mae_list":   val_mae_list,
            "within3_list":   within3_list,
        }, outdir / "checkpoint_last.pth")

    # ── Final evaluation ──
    logger.info("─" * 60)
    logger.info("Loading best model for final evaluation …")
    model.load_state_dict(
        torch.load(outdir / "best_age_efficientnet_b4.pth", map_location=device))
    model.eval()

    within_3 = within_5 = total = 0
    with torch.no_grad():
        for images, ages, _ in val_loader:
            images = images.to(device)
            ages   = ages.to(device)
            pred_ages, _ = model(images, clamp_output=True)
            diff     = torch.abs(pred_ages - ages)
            within_3 += (diff <= 3).sum().item()
            within_5 += (diff <= 5).sum().item()
            total    += ages.size(0)

    logger.info(f"Final best-model  ±3yr: {100*within_3/total:.2f}%  "
                f"±5yr: {100*within_5/total:.2f}%")

    save_plots(train_mae_list, val_mae_list, within3_list, outdir)
    logger.info(f"Plots saved to {outdir}/")
    logger.info("Done.")


if __name__ == "__main__":
    main()
