"""
3D Point Cloud Representation and Provenance Tracking for scene_3d/.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from .camera import PerspectiveCamera3D


@dataclass
class Point3D:
    x: float
    y: float
    z: float
    r: int
    g: int
    b: int
    provenance_code: str = "OBSERVED"  # OBSERVED, DEPTH_INFERRED, GEOMETRY_INFERRED, INPAINTED, LOW_CONFIDENCE
    confidence: float = 1.0


@dataclass
class PointCloud3D:
    vertices: np.ndarray        # (N, 3) float32 [X, Y, Z]
    colors: np.ndarray          # (N, 3) uint8 [R, G, B]
    provenance_labels: List[str] = field(default_factory=list)
    confidence_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    @classmethod
    def from_rgb_depth(
        cls,
        rgb_array: np.ndarray,
        depth_map: np.ndarray,
        camera: PerspectiveCamera3D,
        provenance_map: Optional[np.ndarray] = None
    ) -> 'PointCloud3D':
        h, w, _ = rgb_array.shape
        u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        pts_3d = camera.backproject(u_grid.ravel(), v_grid.ravel(), depth_map.ravel())
        colors = rgb_array.reshape(-1, 3).astype(np.uint8)

        prov_flat = provenance_map.ravel() if provenance_map is not None else np.ones(h * w, dtype=np.float32)
        prov_labels = []
        for p_val in prov_flat:
            if p_val >= 0.95:
                prov_labels.append("OBSERVED")
            elif p_val >= 0.50:
                prov_labels.append("DEPTH_INFERRED")
            else:
                prov_labels.append("INPAINTED")

        return cls(
            vertices=pts_3d,
            colors=colors,
            provenance_labels=prov_labels,
            confidence_scores=prov_flat.astype(np.float32)
        )
