"""
3D Gaussian Splatting Parameter Prediction Scaffold for scene_3d/.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from .camera import PerspectiveCamera3D
from .point_cloud import PointCloud3D


@dataclass
class Gaussian3D:
    position: np.ndarray   # (3,) float32 [x, y, z]
    color: np.ndarray      # (3,) uint8 [r, g, b]
    opacity: float         # float32 [0.0, 1.0]
    scale: np.ndarray      # (3,) float32 [sx, sy, sz]
    rotation: np.ndarray   # (4,) float32 quaternion [qw, qx, qy, qz]
    provenance: str = "OBSERVED"


@dataclass
class GaussianScene:
    gaussians: List[Gaussian3D] = field(default_factory=list)

    @classmethod
    def from_point_cloud(cls, pt_cloud: PointCloud3D, default_scale: float = 0.02) -> 'GaussianScene':
        gaussians = []
        for v, c, prov in zip(pt_cloud.vertices, pt_cloud.colors, pt_cloud.provenance_labels):
            gaussians.append(Gaussian3D(
                position=v,
                color=c,
                opacity=1.0 if prov == "OBSERVED" else 0.8,
                scale=np.array([default_scale, default_scale, default_scale], dtype=np.float32),
                rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                provenance=prov
            ))
        return cls(gaussians=gaussians)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_gaussians": len(self.gaussians),
            "provenance_distribution": {
                "OBSERVED": sum(1 for g in self.gaussians if g.provenance == "OBSERVED"),
                "DEPTH_INFERRED": sum(1 for g in self.gaussians if g.provenance == "DEPTH_INFERRED"),
                "INPAINTED": sum(1 for g in self.gaussians if g.provenance == "INPAINTED")
            }
        }
