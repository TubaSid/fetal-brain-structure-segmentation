"""Loss functions for medical image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, weight=None, ignore_index: int = -1):
        """
        Args:
            alpha: Weighting factor in range (0,1) to balance classes
            gamma: Exponent of the modulating factor (1 - p_t) ^ gamma
            weight: Manual rescaling weight for each class
            ignore_index: Specifies a target value to ignore
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
    
    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, H, W) model output
            target: (B, H, W) target labels
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(
            logits, target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss for segmentation."""
    
    def __init__(self, n_classes: int, class_weights: torch.Tensor = None, eps: float = 1e-6):
        """
        Args:
            n_classes: Number of classes
            class_weights: Per-class weights
            eps: Small value for numerical stability
        """
        super().__init__()
        self.n_classes = n_classes
        self.eps = eps
        
        if class_weights is None:
            class_weights = torch.ones(n_classes, dtype=torch.float32)
        self.register_buffer('w', class_weights / class_weights.mean())
    
    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, H, W) model output
            target: (B, H, W) target labels
            
        Returns:
            Dice loss value
        """
        probs = torch.softmax(logits, dim=1)
        target_1h = F.one_hot(
            torch.clamp(target, 0, self.n_classes - 1),
            num_classes=self.n_classes
        ).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        numerator = 2 * (probs * target_1h).sum(dims) + self.eps
        denominator = probs.pow(2).sum(dims) + target_1h.pow(2).sum(dims) + self.eps
        
        dice_c = numerator / denominator
        return 1 - (dice_c * self.w).sum() / self.w.sum()


def combined_loss(
    logits,
    target,
    ce_loss_fn,
    focal_loss_fn,
    dice_loss_fn,
    dice_weight: float = 0.4,
    focal_weight: float = 0.2
):
    """
    Combined loss: CE + Focal + Dice.
    
    Args:
        logits: Model output
        target: Target labels
        ce_loss_fn: Cross-entropy loss function
        focal_loss_fn: Focal loss function
        dice_loss_fn: Dice loss function
        dice_weight: Weight for dice loss
        focal_weight: Weight for focal loss
        
    Returns:
        Combined loss value
    """
    ce_loss = ce_loss_fn(logits, target)
    focal_loss = focal_loss_fn(logits, target)
    dice_loss = dice_loss_fn(logits, target)
    
    ce_weight = 1 - dice_weight - focal_weight
    return ce_weight * ce_loss + focal_weight * focal_loss + dice_weight * dice_loss
