"""
Edge Analysis Pipeline and Structural Boundary Map Generation.
Fused multi-modal edge detection combining Sobel, Canny, Laplacian, RGB gradients, and Depth gradients.
"""

from typing import Dict, Tuple, Optional
import cv2
import numpy as np


def compute_edge_maps(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    subject_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Computes a comprehensive suite of structural edge diagnostic maps:
    1. Sobel X & Y gradients
    2. Gradient Magnitude
    3. Canny Edge Map
    4. Laplacian Edge Map
    5. Depth Gradient Map
    6. Structural Fused Edge Map
    7. Edge Confidence Map
    """
    h, w, _ = rgb_array.shape
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    # 1. Sobel X and Y
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    max_sobel = sobel_mag.max() if sobel_mag.max() > 0 else 1.0
    sobel_norm = (sobel_mag / max_sobel).astype(np.float32)

    # 2. Canny Edge Map
    canny = (cv2.Canny(gray, 40, 120) > 0).astype(np.float32)

    # 3. Laplacian Edge Map
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    max_lap = abs(laplacian).max() if abs(laplacian).max() > 0 else 1.0
    lap_norm = (abs(laplacian) / max_lap).astype(np.float32)

    # 4. Depth Gradient Map
    d_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    d_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    d_mag = np.sqrt(d_x**2 + d_y**2)
    max_d = d_mag.max() if d_mag.max() > 0 else 1.0
    depth_grad_norm = (d_mag / max_d).astype(np.float32)

    # 5. Subject Boundary Edge Map (if mask provided)
    if subject_mask is not None:
        sub_boundary = (cv2.Canny((subject_mask * 255).astype(np.uint8), 100, 200) > 0).astype(np.float32)
    else:
        sub_boundary = np.zeros((h, w), dtype=np.float32)

    # 6. Fused Edge Map (RGB + Depth + Subject Boundary)
    fused = 0.35 * sobel_norm + 0.25 * canny + 0.20 * depth_grad_norm + 0.20 * sub_boundary
    fused_norm = np.clip(fused, 0.0, 1.0).astype(np.float32)

    # 7. Edge Confidence Map
    edge_conf = cv2.GaussianBlur(fused_norm, (5, 5), 0)
    edge_conf_norm = np.clip(edge_conf, 0.0, 1.0).astype(np.float32)

    return {
        "sobel_mag": sobel_norm,
        "canny": canny,
        "laplacian": lap_norm,
        "depth_grad": depth_grad_norm,
        "fused": fused_norm,
        "edge_confidence": edge_conf_norm
    }
