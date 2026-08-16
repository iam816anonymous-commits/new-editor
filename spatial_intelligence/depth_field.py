"""
Structured 2.5D Depth Field and Multi-Layer Spatial Depth for Spatial Intelligence.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from .schemas import DepthField, Entity, SceneGraph


def build_spatial_depth_field(
    refined_depth: np.ndarray,
    background_depth: np.ndarray,
    subject_mask: np.ndarray,
    confidence_map: np.ndarray,
    provenance_map: np.ndarray,
    scene_graph: Optional[SceneGraph] = None
) -> DepthField:
    """
    Constructs a structured 2.5D DepthField separating background depth, subject depth,
    internal subject depth variation, and uncertainty bounds while preserving provenance.
    """
    h, w = refined_depth.shape
    d_min, d_max = float(refined_depth.min()), float(refined_depth.max())

    # Subject depth field: subject pixels keep refined depth, non-subject pixels are set to max background depth
    subject_depth = np.full((h, w), fill_value=d_max, dtype=np.float32)
    subject_depth[subject_mask] = refined_depth[subject_mask]

    # Calculate internal subject spatial layering if scene graph with entities/parts is available
    rendering_depth = refined_depth.copy()

    if scene_graph is not None and scene_graph.entities:
        for ent in scene_graph.entities.values():
            if ent.is_primary_subject and ent.parts:
                # Modulate internal rendering depth across sub-parts for subtle multi-layer internal parallax
                for part in ent.parts:
                    p_mask = part.mask
                    if np.any(p_mask):
                        # Ensure part depth remains continuous and strictly within valid range
                        p_depth_mean = np.mean(refined_depth[p_mask])
                        rendering_depth[p_mask] = np.clip(0.95 * refined_depth[p_mask] + 0.05 * p_depth_mean, d_min, d_max)

    # Depth uncertainty map: inverse of gradient-edge confidence map plus region boundaries
    depth_grad_x = cv2.Sobel(refined_depth, cv2.CV_32F, 1, 0, ksize=3)
    depth_grad_y = cv2.Sobel(refined_depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_grad_mag = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
    norm_grad_mag = np.clip(depth_grad_mag / max(1e-5, float(depth_grad_mag.max())), 0.0, 1.0)

    uncertainty_map = np.clip((1.0 - confidence_map) * 0.6 + norm_grad_mag * 0.4, 0.0, 1.0).astype(np.float32)

    return DepthField(
        rendering_depth=rendering_depth,
        background_depth=background_depth,
        subject_depth=subject_depth,
        uncertainty_map=uncertainty_map,
        provenance_map=provenance_map,
        min_depth=d_min,
        max_depth=d_max
    )
