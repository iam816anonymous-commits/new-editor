"""
Canonical Perspective Camera Abstraction for scene_3d/.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class PerspectiveCamera3D:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def create_canonical(cls, width: int, height: int) -> 'PerspectiveCamera3D':
        focal = float(max(width, height))
        return cls(
            width=width,
            height=height,
            fx=focal,
            fy=focal,
            cx=width / 2.0,
            cy=height / 2.0
        )

    def backproject(self, u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Backprojects 2D image coordinates (u, v) and depth Z to 3D points [X, Y, Z]."""
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return np.column_stack([X, Y, Z]).astype(np.float32)

    def project(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Projects 3D points [X, Y, Z] to 2D image coordinates (u, v, Z)."""
        Z = np.where(np.abs(points_3d[:, 2]) < 1e-6, 1e-6, points_3d[:, 2])
        u = self.fx * (points_3d[:, 0] / Z) + self.cx
        v = self.fy * (points_3d[:, 1] / Z) + self.cy
        return u, v, Z
