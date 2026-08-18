"""
Mask Merging and Refinement for Semantic Subject Selection Module.
"""

import cv2
import numpy as np
from typing import List


def merge_candidate_masks(masks: List[np.ndarray]) -> np.ndarray:
    """Computes the boolean union of multiple candidate masks."""
    if not masks:
        raise ValueError("Cannot merge empty list of masks.")
    merged = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        merged |= m
    return merged


def refine_subject_mask(
    mask_bool: np.ndarray,
    rgb_array: np.ndarray,
    min_component_area_ratio: float = 0.001,
    close_kernel_size: int = 5
) -> np.ndarray:
    """
    Refines merged subject mask:
    1. Removes tiny isolated noise components (< min_component_area_ratio of total image).
    2. Fills internal holes in the primary subject.
    3. Applies controlled morphological closing to smooth small cracks.
    4. Applies RGB edge guidance to preserve thin structures (hair, fingers, ornaments, snake heads).
    """
    h, w = mask_bool.shape
    total_pixels = max(1, h * w)
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255

    # 1. Filter out tiny isolated noise components
    min_pixels = int(min_component_area_ratio * total_pixels)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    filtered_uint8 = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            filtered_uint8[labels == i] = 255

    if np.sum(filtered_uint8) == 0:
        filtered_uint8 = mask_uint8.copy()

    # 2. Fill internal holes
    contours, hierarchy = cv2.findContours(filtered_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_uint8 = np.zeros_like(filtered_uint8)
    if contours:
        cv2.drawContours(filled_uint8, contours, -1, 255, thickness=cv2.FILLED)
    else:
        filled_uint8 = filtered_uint8.copy()

    # 3. Morphological closing with ellipse kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    closed_uint8 = cv2.morphologyEx(filled_uint8, cv2.MORPH_CLOSE, kernel)

    # 4. RGB edge-guided boundary protection (do not bleed across sharp RGB edges)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edge_mask = (edges > 0)

    # Re-apply edge restriction on closed expansion
    refined_uint8 = closed_uint8.copy()
    expansion_pixels = (closed_uint8 > 0) & (~(filtered_uint8 > 0))
    refined_uint8[expansion_pixels & edge_mask] = 0

    return (refined_uint8 > 0)
