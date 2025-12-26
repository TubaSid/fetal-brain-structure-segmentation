#!/usr/bin/env python
"""
Training script for fetal brain segmentation model.

Usage:
    python train.py --config configs/default_config.yaml
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    Config,
    SegDataset,
    list_pairs,
    build_transforms,
    UNet,
    FocalLoss,
    SoftDiceLoss,
    combined_loss,
    confusion_matrix,
    iou_from_cm,
    dice_from_cm,
    metric_score,
)


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_histogram(dataset: SegDataset, n_classes: int) -> np.ndarray:
    """Compute class frequency in dataset."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for _, mask_path in dataset.pairs:
        import cv2
        from src.data import color_mask_to_index
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        idx = color_mask_to_index(m)
        for c in range(n_classes):
            counts[c] += (idx == c).sum()
    return counts


def load_config_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config file into a dict of overrides. Returns empty dict if not found."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required to load --config. Install with 'pip install pyyaml'.") from e
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must contain a mapping (key: value) at top-level.")
    return data


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device,
    cfg: Config,
    scaler=None,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc="Train")
    for img, mask, _ in pbar:
        img = img.to(device, non_blocking=True).float()
        mask = mask.to(device, non_blocking=True).long()
        
        optimizer.zero_grad()
        
        if cfg.amp and scaler is not None:
            with autocast():
                logits = model(img)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(img)
            loss = criterion(logits, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device,
    cfg: Config,
):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_cms = []
    
    pbar = tqdm(loader, desc="Val")
    for img, mask, _ in pbar:
        img = img.to(device, non_blocking=True).float()
        mask = mask.to(device, non_blocking=True).long()
        
        logits = model(img)
        loss = criterion(logits, mask)
        
        pred = torch.argmax(logits, dim=1)
        cm = confusion_matrix(pred, mask, cfg.n_classes)
        all_cms.append(cm)
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Aggregate metrics
    total_cm = sum(all_cms).float()
    val_loss = total_loss / len(loader)
    val_iou = iou_from_cm(total_cm)
    val_dice = dice_from_cm(total_cm)
    
    return val_loss, val_iou, val_dice


def main(args):
    """Main training loop."""
    seed_everything(42)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = Config()
    # Optional: override with YAML
    overrides = load_config_yaml(getattr(args, 'config', None))
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Config: {cfg}")
    
    # Data
    img_dir = os.path.join(cfg.DATA_ROOT, cfg.IMG_DIR)
    mask_dir = os.path.join(cfg.DATA_ROOT, cfg.MASK_DIR)
    
    pairs = list_pairs(img_dir, mask_dir)
    print(f"Found {len(pairs)} image-mask pairs")
    
    # Train/val split
    random.shuffle(pairs)
    split = int(0.85 * len(pairs))
    pairs_train, pairs_val = pairs[:split], pairs[split:]
    print(f"Train: {len(pairs_train)} | Val: {len(pairs_val)}")
    
    # Datasets
    train_tf = build_transforms(cfg.img_size, is_train=True)
    val_tf = build_transforms(cfg.img_size, is_train=False)
    ds_train = SegDataset(pairs_train, transform=train_tf)
    ds_val = SegDataset(pairs_val, transform=val_tf)
    
    # Class weights
    class_counts = compute_class_histogram(ds_train, cfg.n_classes)
    class_freq = class_counts / np.maximum(class_counts.sum(), 1)
    inv_freq = 1.0 / np.clip(class_freq, 1e-8, None)
    inv_freq = inv_freq / inv_freq.mean()
    ce_weights = (1 - cfg.ce_weight_smoothing) * inv_freq + cfg.ce_weight_smoothing * np.ones_like(inv_freq)
    print(f"Class counts: {class_counts}")
    print(f"CE weights: {ce_weights}")
    
    # Oversampling
    sample_weights = [cfg.oversample_factor if has else 1.0 for has in ds_train.contains_small]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # DataLoaders
    loader_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Model
    model = UNet(in_channels=cfg.in_channels, n_classes=cfg.n_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Loss functions
    ce_weight_t = torch.tensor(ce_weights, dtype=torch.float32, device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=ce_weight_t, ignore_index=-1)
    focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0, weight=ce_weight_t)
    dice_w = torch.tensor([0.5, 1.0, 2.0, 2.0], dtype=torch.float32, device=device)
    dice_loss_fn = SoftDiceLoss(cfg.n_classes, class_weights=dice_w)
    
    def criterion(logits, target):
        return combined_loss(
            logits, target,
            ce_loss_fn, focal_loss_fn, dice_loss_fn,
            dice_weight=cfg.dice_weight,
            focal_weight=cfg.focal_weight
        )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # AMP
    scaler = GradScaler() if cfg.amp else None
    
    # Training
    best_score = -np.inf
    best_path = os.path.join(cfg.OUT_DIR, 'best_unet.pt')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': [],
        'lr': [],
    }
    
    for epoch in range(cfg.epochs):
        t0 = time.time()
        
        # Train
        train_loss = train_epoch(model, loader_train, criterion, optimizer, device, cfg, scaler)
        
        # Validate
        val_loss, val_iou, val_dice = validate(model, loader_val, criterion, device, cfg)
        
        # Scheduler
        scheduler.step(val_loss)
        curr_lr = optimizer.param_groups[0]['lr']
        
        # Checkpointing
        score = metric_score(val_iou, cfg.ckpt_metric_weights)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou.tolist())
        history['val_dice'].append(val_dice.tolist())
        history['lr'].append(curr_lr)
        
        dt = time.time() - t0
        print(
            f"[{epoch:03d}/{cfg.epochs}] "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"IoU: {val_iou.round(3)} | score={score:.4f} | "
            f"lr={curr_lr:.2e} | {dt:.1f}s"
        )
        
        if score > best_score:
            best_score = score
            torch.save({
                'model': model.state_dict(),
                'cfg': cfg.__dict__,
                'score': best_score,
                'epoch': epoch,
            }, best_path)
            print(f"  -> saved new best to {best_path} (score={best_score:.4f})")
    
    print(f"\nTraining complete! Best model: {best_path} (score={best_score:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train fetal brain segmentation model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to YAML config file (optional)')
    args = parser.parse_args()
    
    main(args)
