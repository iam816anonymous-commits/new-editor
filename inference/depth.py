import torch
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from inference.device import get_device

def infer_raw_depth(
    pil_img: Image.Image,
    processor: AutoImageProcessor,
    model: AutoModelForDepthEstimation,
    device: Optional[str] = None
) -> np.ndarray:
    """Performs real Depth Anything V2 inference on the image and returns a 2D raw depth array."""
    if device is None:
        device = get_device()

    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original image resolution
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=pil_img.size[::-1],
        mode="bicubic",
        align_corners=False
    )
    raw_depth = prediction.squeeze().cpu().numpy()

    if raw_depth.ndim != 2:
        raise ValueError(f"Expected 2D raw depth array, got shape {raw_depth.shape}")
    if np.isnan(raw_depth).any() or np.isinf(raw_depth).any():
        raise ValueError("Raw depth contains NaN or Inf values.")

    return raw_depth

def handle_depth_outliers_and_normalize(
    raw_depth: np.ndarray,
    p_min: float = 1.0,
    p_max: float = 99.0,
    target_min: float = 0.1,
    target_max: float = 10.0
) -> np.ndarray:
    """
    Handles depth outliers using percentile clipping and normalizes relative monocular depth to
    normalized scene depth / rendering coordinates Z in range [target_min, target_max].
    Note: Monocular depth is relative, NOT true physical metric depth in meters.
    Depth Anything V2 outputs relative disparity (higher values = closer to camera).
    We convert high disparity -> closer Z (small coordinate value) and low disparity -> farther Z (large coordinate value).
    """
    p_low = np.percentile(raw_depth, p_min)
    p_high = np.percentile(raw_depth, p_max)

    if p_high <= p_low:
        p_high = p_low + 1e-6

    clipped_depth = np.clip(raw_depth, p_low, p_high)

    # Min-max scaling to [0, 1]
    norm_0_1 = (clipped_depth - p_low) / (p_high - p_low)

    # Invert so 1.0 (closest) maps to target_min and 0.0 (farthest) maps to target_max
    rendering_depth = target_max - norm_0_1 * (target_max - target_min)

    return rendering_depth.astype(np.float32)

def edge_aware_depth_refinement(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    d: int = 9,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0
) -> np.ndarray:
    """
    Refines depth map to create RENDERING_DEPTH from RAW_DEPTH:
    1. Applies RGB-guided bilateral filtering to smooth small intra-surface depth noise.
    2. Preserves sharp true depth discontinuities at Canny RGB boundaries.
    """
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max <= d_min:
        return depth_map.copy()

    depth_norm = ((depth_map - d_min) / max(1e-5, d_max - d_min) * 255.0).astype(np.uint8)

    # Bilateral filter on normalized depth map
    filtered_norm = cv2.bilateralFilter(
        depth_norm,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )

    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = (edges > 0)

    # Do not cross-smooth across Canny RGB edges
    refined_norm = filtered_norm.copy()
    refined_norm[edge_mask] = depth_norm[edge_mask]

    refined_depth = d_min + (refined_norm.astype(np.float32) / 255.0) * (d_max - d_min)
    return refined_depth.astype(np.float32)

def compute_depth_confidence_map(
    depth_map: np.ndarray,
    rgb_array: np.ndarray
) -> np.ndarray:
    """
    Computes a depth confidence map in range [0.0, 1.0].
    Measures depth gradient alignment with RGB edges to detect edge ambiguity/uncertainty.
    Higher values indicate higher confidence.
    """
    # Compute depth gradients
    depth_grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    depth_grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    depth_grad_mag = np.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2)

    # Compute RGB intensity gradients
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rgb_grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    rgb_grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    rgb_grad_mag = np.sqrt(rgb_grad_x ** 2 + rgb_grad_y ** 2)

    # Normalize magnitudes
    max_d_grad = depth_grad_mag.max() if depth_grad_mag.max() > 0 else 1.0
    max_c_grad = rgb_grad_mag.max() if rgb_grad_mag.max() > 0 else 1.0

    norm_d_grad = depth_grad_mag / max_d_grad
    norm_c_grad = rgb_grad_mag / max_c_grad

    # In regions where depth has strong gradients but RGB has no edge, confidence is lower
    unexplained_depth_edges = np.clip(norm_d_grad - norm_c_grad, 0.0, 1.0)
    confidence = 1.0 - 0.5 * unexplained_depth_edges

    return confidence.astype(np.float32)
