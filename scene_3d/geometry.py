"""
Depth-Aware Surface Mesh Geometry Triangulation for scene_3d/.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from .camera import PerspectiveCamera3D
from .point_cloud import PointCloud3D


@dataclass
class MeshGeometry3D:
    vertices: np.ndarray  # (N, 3) float32
    colors: np.ndarray    # (N, 3) uint8
    faces: np.ndarray     # (M, 3) int32
    provenance_labels: List[str] = field(default_factory=list)


def construct_depth_mesh(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    camera: PerspectiveCamera3D,
    provenance_map: Optional[np.ndarray] = None,
    depth_discontinuity_threshold: float = 0.5
) -> MeshGeometry3D:
    """
    Constructs triangulated 3D mesh, breaking quad connections across sharp depth jumps.
    """
    h, w, _ = rgb_array.shape
    pt_cloud = PointCloud3D.from_rgb_depth(rgb_array, depth_map, camera, provenance_map=provenance_map)

    indices = np.arange(h * w, dtype=np.int32).reshape(h, w)
    i00 = indices[:-1, :-1].ravel()
    i10 = indices[:-1, 1:].ravel()
    i01 = indices[1:, :-1].ravel()
    i11 = indices[1:, 1:].ravel()

    z00 = depth_map[:-1, :-1].ravel()
    z10 = depth_map[:-1, 1:].ravel()
    z01 = depth_map[1:, :-1].ravel()
    z11 = depth_map[1:, 1:].ravel()

    d1 = np.maximum(np.abs(z00 - z10), np.abs(z00 - z01))
    d2 = np.maximum(np.abs(z11 - z10), np.abs(z11 - z01))

    valid_quads = (d1 < depth_discontinuity_threshold) & (d2 < depth_discontinuity_threshold)

    f1 = np.column_stack([i00[valid_quads], i10[valid_quads], i01[valid_quads]])
    f2 = np.column_stack([i10[valid_quads], i11[valid_quads], i01[valid_quads]])
    faces = np.vstack([f1, f2]).astype(np.int32)

    return MeshGeometry3D(
        vertices=pt_cloud.vertices,
        colors=pt_cloud.colors,
        faces=faces,
        provenance_labels=pt_cloud.provenance_labels
    )
