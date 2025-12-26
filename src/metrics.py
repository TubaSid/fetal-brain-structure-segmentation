"""Metrics for segmentation evaluation."""

import torch
import numpy as np


@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Compute confusion matrix.
    
    Args:
        pred: (B, H, W) predicted class indices
        target: (B, H, W) target class indices
        n_classes: Number of classes
        
    Returns:
        (n_classes, n_classes) confusion matrix
    """
    k = (target >= 0) & (target < n_classes)
    bins = target[k].to(torch.int64) * n_classes + pred[k].to(torch.int64)
    cm = torch.bincount(bins, minlength=n_classes**2).reshape(n_classes, n_classes)
    return cm


def iou_from_cm(cm: torch.Tensor) -> np.ndarray:
    """
    Compute IoU per class from confusion matrix.
    
    Args:
        cm: (n_classes, n_classes) confusion matrix
        
    Returns:
        (n_classes,) IoU per class
    """
    tp = cm.diagonal()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = tp + fp + fn
    iou = (tp / torch.clamp(denom, min=1)).cpu().numpy()
    return iou


def dice_from_cm(cm: torch.Tensor) -> np.ndarray:
    """
    Compute Dice per class from confusion matrix.
    
    Args:
        cm: (n_classes, n_classes) confusion matrix
        
    Returns:
        (n_classes,) Dice per class
    """
    tp = cm.diagonal()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    dice = (2 * tp / torch.clamp(2 * tp + fp + fn, min=1)).cpu().numpy()
    return dice


def metric_score(iou: np.ndarray, weights: dict) -> float:
    """
    Compute weighted metric for model selection.
    
    Args:
        iou: Per-class IoU
        weights: Dict mapping class index to weight
        
    Returns:
        Weighted score
    """
    num = sum(iou[c] * w for c, w in weights.items())
    denom = sum(w for w in weights.values())
    return float(num / max(denom, 1e-8))
