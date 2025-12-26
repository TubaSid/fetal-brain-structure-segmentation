"""Post-processing utilities for segmentation predictions."""

import cv2
import numpy as np


def _keep_largest(bin_mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), 8)
    if n <= 1:
        return bin_mask.astype(bool)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def postprocess_pred(
    pred: np.ndarray,
    dilate_px: int = 10,
    par_kernel: int = 7,
    small_min_area: int = 15,
    small_keep_k: int = 2
) -> np.ndarray:
    """
    Post-process predictions: clean parenchyma, constrain CSP/LV.
    
    Args:
        pred: (H, W) class indices
        dilate_px: Dilation size for parenchyma
        par_kernel: Kernel size for parenchyma morphology
        small_min_area: Min area for CSP/LV components
        small_keep_k: Max number of CSP/LV components to keep
        
    Returns:
        Post-processed prediction
    """
    out = pred.copy()
    
    # Clean parenchyma (class 1)
    par = (out == 1).astype(np.uint8)
    par = cv2.morphologyEx(
        par, cv2.MORPH_CLOSE,
        np.ones((par_kernel, par_kernel), np.uint8)
    )
    par = _keep_largest(par).astype(np.uint8)
    out[out == 1] = 0
    out[par > 0] = 1
    
    # Constrain CSP/LV (classes 2 and 3) to parenchyma region
    par_dil = cv2.dilate(
        par, np.ones((dilate_px, dilate_px), np.uint8), 1
    ).astype(bool)
    
    for class_idx in (2, 3):
        cls = (out == class_idx) & par_dil
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cls.astype(np.uint8), 8)
        
        keep = np.zeros_like(cls, dtype=bool)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            order = np.argsort(-areas)
            kept = 0
            for j in order:
                if areas[j] < small_min_area:
                    break
                keep |= (labels == (j + 1))
                kept += 1
                if small_keep_k is not None and kept >= small_keep_k:
                    break
        
        out[out == class_idx] = 0
        out[keep] = class_idx
    
    return out


def refine_lv_safe(
    pred: np.ndarray,
    keep_k: int = 2,
    min_area: int = 5,
    morph: str = 'close',
    ksize: int = 3
) -> np.ndarray:
    """
    Refine LV (lateral ventricle) predictions.
    
    Args:
        pred: (H, W) class indices
        keep_k: Max number of LV components to keep
        min_area: Min area for LV components
        morph: Morphological operation ('close', 'open')
        ksize: Kernel size for morphological operation
        
    Returns:
        Boolean mask for LV class
    """
    lv = (pred == 3).astype(np.uint8)
    
    if morph == 'close':
        lv = cv2.morphologyEx(
            lv, cv2.MORPH_CLOSE,
            np.ones((ksize, ksize), np.uint8)
        )
    elif morph == 'open':
        lv = cv2.morphologyEx(
            lv, cv2.MORPH_OPEN,
            np.ones((ksize, ksize), np.uint8)
        )
    
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lv, 8)
    if n <= 1:
        return (pred == 3)
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    comp_ids = np.arange(1, n)
    ok = areas >= min_area
    keep_ids = comp_ids[ok][:keep_k]
    keep = np.isin(labels, keep_ids)
    
    return keep if keep.any() else (pred == 3)
