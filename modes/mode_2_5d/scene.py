"""
2.5D Layered Scene Representation and SceneLayer Data Model.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class SceneLayer:
    """Individual renderable scene layer in LayeredScene."""
    layer_id: str
    semantic_role: str  # BACKGROUND, MIDGROUND, PRIMARY_SUBJECT, FOREGROUND
    rgb: np.ndarray
    alpha: np.ndarray
    depth: np.ndarray
    valid_mask: np.ndarray
    inpaint_mask: np.ndarray
    z_min: float
    z_max: float
    priority: int

@dataclass
class LayeredScene:
    """Persistent neural/layered 2.5D scene representation."""
    layers: List[SceneLayer]
    source_image: np.ndarray
    refined_depth: np.ndarray
    subject_mask: np.ndarray
    occlusion_edge_map: np.ndarray
    convergence_depth: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Scene25D:
    """Legacy Dataclass holding 2.5D layered scene components."""
    rgb_array: np.ndarray
    depth_map: np.ndarray
    subject_mask: np.ndarray
    bg_plate: np.ndarray
    bg_depth: np.ndarray
    confidence_map: np.ndarray


def detect_occlusion_boundaries(depth: np.ndarray, threshold_ratio: float = 0.08) -> np.ndarray:
    """
    Detect depth discontinuity / occlusion boundaries using normalized Sobel gradient analysis.
    Returns binary mask (uint8 0 or 255) of occlusion edges.
    """
    import cv2
    gx = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    depth_range = float(np.ptp(depth)) if np.ptp(depth) > 1e-6 else 1.0
    threshold = threshold_ratio * depth_range
    edges = (grad_mag > threshold).astype(np.uint8) * 255
    return edges


def construct_layered_scene(
    source_rgb: np.ndarray,
    depth_map: np.ndarray,
    subject_mask: Optional[np.ndarray] = None,
    bg_plate: Optional[np.ndarray] = None,
    bg_depth: Optional[np.ndarray] = None,
) -> LayeredScene:
    """Constructs a LayeredScene object with structured SceneLayers and occlusion boundary edges."""
    h, w = source_rgb.shape[:2]
    if subject_mask is None:
        subj_mask_bool = np.zeros((h, w), dtype=bool)
    else:
        subj_mask_bool = subject_mask.astype(bool)

    if bg_plate is None:
        bg_plate = source_rgb.copy()
    if bg_depth is None:
        bg_depth = depth_map.copy()

    occlusion_edges = detect_occlusion_boundaries(depth_map)

    if np.any(subj_mask_bool):
        convergence_depth = float(np.median(depth_map[subj_mask_bool]))
    else:
        convergence_depth = float(np.median(depth_map))

    layers: List[SceneLayer] = []

    # 1. Background layer
    bg_mask = ~subj_mask_bool
    bg_z_min = float(np.min(bg_depth)) if bg_depth.size > 0 else 1.0
    bg_z_max = float(np.max(bg_depth)) if bg_depth.size > 0 else 10.0
    layers.append(SceneLayer(
        layer_id="background",
        semantic_role="BACKGROUND",
        rgb=bg_plate,
        alpha=np.ones((h, w), dtype=np.float32),
        depth=bg_depth,
        valid_mask=np.ones((h, w), dtype=bool),
        inpaint_mask=(occlusion_edges > 0) & bg_mask,
        z_min=bg_z_min,
        z_max=bg_z_max,
        priority=0
    ))

    # 2. Subject layer (if present)
    if np.any(subj_mask_bool):
        subj_rgb = np.zeros_like(source_rgb)
        subj_rgb[subj_mask_bool] = source_rgb[subj_mask_bool]
        subj_depth = depth_map.copy()
        subj_z_min = float(np.min(depth_map[subj_mask_bool]))
        subj_z_max = float(np.max(depth_map[subj_mask_bool]))
        layers.append(SceneLayer(
            layer_id="primary_subject",
            semantic_role="PRIMARY_SUBJECT",
            rgb=subj_rgb,
            alpha=subj_mask_bool.astype(np.float32),
            depth=subj_depth,
            valid_mask=subj_mask_bool,
            inpaint_mask=np.zeros((h, w), dtype=bool),
            z_min=subj_z_min,
            z_max=subj_z_max,
            priority=1
        ))

    return LayeredScene(
        layers=layers,
        source_image=source_rgb,
        refined_depth=depth_map,
        subject_mask=subj_mask_bool,
        occlusion_edge_map=occlusion_edges,
        convergence_depth=convergence_depth,
        metadata={"layer_count": len(layers)}
    )
