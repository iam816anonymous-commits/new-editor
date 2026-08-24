"""
Inferred 3D Scene Rasterizer and Novel View Renderer for scene_3d/.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from .camera import PerspectiveCamera3D
from .point_cloud import PointCloud3D
from .geometry import MeshGeometry3D


class Inferred3DRenderer:
    """
    Renders novel views from explicit 3D PointCloud3D / MeshGeometry3D scene representations.
    """

    @staticmethod
    def render_point_cloud_view(
        pt_cloud: PointCloud3D,
        camera: PerspectiveCamera3D,
        R_mat: np.ndarray,
        t_vec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Renders novel view from 3D point cloud via SE(3) transformation P' = P @ R.T + t and Z-buffering.
        Returns: (rendered_rgb, z_buffer)
        """
        h, w = camera.height, camera.width
        z_buf = np.full((h, w), fill_value=1e9, dtype=np.float32)
        syn_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # SE(3) Transformation
        pts_trans = (pt_cloud.vertices @ R_mat.T) + t_vec
        u_proj, v_proj, z_proj = camera.project(pts_trans)

        valid = (z_proj > 0.05) & (u_proj >= 0) & (u_proj < w - 1) & (v_proj >= 0) & (v_proj < h - 1)
        if not np.any(valid):
            return syn_rgb, z_buf

        pu = np.floor(u_proj[valid]).astype(int)
        pv = np.floor(v_proj[valid]).astype(int)
        pz = z_proj[valid]
        pcol = pt_cloud.colors[valid]

        sort_idx = np.argsort(-pz)
        pu, pv, pz, pcol = pu[sort_idx], pv[sort_idx], pz[sort_idx], pcol[sort_idx]

        for u, v, z, col in zip(pu, pv, pz, pcol):
            if z < z_buf[v, u]:
                z_buf[v, u] = z
                syn_rgb[v, u] = col

        return syn_rgb, z_buf
