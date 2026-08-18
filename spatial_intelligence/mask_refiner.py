"""
Edge-Snapped Mask Refinement and Topological Mask Cleaning.
Snaps segmentation mask contours toward strong RGB/depth gradients while preserving topology and applying role-aware feathering.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def snap_mask_to_edges(
    mask: np.ndarray,
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    max_snap_distance: int = 4
) -> Tuple[np.ndarray, float, float]:
    """
    Refines mask contour by snapping boundary pixels toward strong nearby RGB and depth gradients.

    Returns:
        (refined_mask, avg_displacement_px, max_displacement_px)
    """
    h, w = mask.shape
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    # Compute RGB and depth edges
    rgb_edges = cv2.Canny(gray, 40, 120) > 0
    d_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    d_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    d_edges = np.sqrt(d_x**2 + d_y**2) > 0.5

    fused_edges = (rgb_edges | d_edges).astype(np.uint8)

    # Compute distance transform from strong edges
    dist_to_edge, labels = cv2.distanceTransformWithLabels(
        1 - fused_edges, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )

    # Extract mask contour boundary
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    refined_mask = mask.copy()
    displacements = []

    for cnt in contours:
        for pt in cnt:
            px, py = pt[0][0], pt[0][1]
            if 0 <= py < h and 0 <= px < w:
                d = dist_to_edge[py, px]
                if 0 < d <= max_snap_distance:
                    displacements.append(float(d))

    avg_disp = float(np.mean(displacements)) if displacements else 0.0
    max_disp = float(np.max(displacements)) if displacements else 0.0

    # Apply guided morphological snap refinement within snap distance
    snap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_snap_distance, max_snap_distance))
    dilated_mask = cv2.dilate(mask_uint8, snap_kernel) > 0
    eroded_mask = cv2.erode(mask_uint8, snap_kernel) > 0

    # Snapped mask takes edge pixels in transition zone
    transition_zone = dilated_mask & (~eroded_mask)
    refined_mask[transition_zone & (fused_edges > 0)] = True

    # Topological cleaning: morphological closing & opening
    clean_uint8 = cv2.morphologyEx((refined_mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    clean_uint8 = cv2.morphologyEx(clean_uint8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    return clean_uint8 > 0, avg_disp, max_disp


def compute_role_aware_alpha_feather(
    mask: np.ndarray,
    layer_role_str: str = "PRIMARY_SUBJECT",
    feather_px: int = 3
) -> np.ndarray:
    """
    Applies role-aware controlled alpha feathering:
    - Hard interior core retained at alpha = 1.0.
    - Narrow transition zone (2-4px) blurred for seamless compositing.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    role_upper = layer_role_str.upper()

    if role_upper in ["PRIMARY_SUBJECT", "PRIMARY_SUBJECT_PART"]:
        blur_k = max(3, feather_px if feather_px % 2 == 1 else feather_px + 1)
    elif role_upper == "FOREGROUND":
        blur_k = max(5, feather_px + 2 if (feather_px + 2) % 2 == 1 else feather_px + 3)
    else:
        blur_k = max(7, feather_px + 4 if (feather_px + 4) % 2 == 1 else feather_px + 5)

    blurred = cv2.GaussianBlur(mask_uint8.astype(np.float32), (blur_k, blur_k), 0) / 255.0

    # Protect interior core
    if role_upper in ["PRIMARY_SUBJECT", "PRIMARY_SUBJECT_PART"]:
        core = cv2.erode(mask_uint8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        blurred[core] = 1.0

    return np.clip(blurred, 0.0, 1.0).astype(np.float32)
