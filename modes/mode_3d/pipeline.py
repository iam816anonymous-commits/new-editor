"""
Mode B (Inferred 3D Scene) Pipeline Implementation.
"""

from typing import Dict, Any
import numpy as np
from core.contracts import RenderRequest
from scene_3d.scene import Inferred3DScene
from scene_3d.camera import PerspectiveCamera3D
from scene_3d.point_cloud import PointCloud3D
from scene_3d.geometry import MeshGeometry3D

class Mode3DPipeline:
    """Standalone Inferred 3D Scene Pipeline Execution Engine."""

    def __init__(self, request: RenderRequest):
        self.request = request

    def execute(self, rgb_array: np.ndarray, depth_map: np.ndarray, subject_mask: np.ndarray) -> Dict[str, Any]:
        """Constructs 3D scene representation and renders viewpoint sweep."""
        h, w, _ = rgb_array.shape
        camera = PerspectiveCamera3D(width=w, height=h, focal_length_x=float(w), focal_length_y=float(w), principal_point_x=w/2.0, principal_point_y=h/2.0)
        scene = Inferred3DScene(rgb_array=rgb_array, depth_map=depth_map, subject_mask=subject_mask, camera=camera)

        # Render keyframe viewpoints
        rendered_frames = []
        for yaw in np.linspace(-15.0, 15.0, num=self.request.frame_count):
            frame = scene.render_viewpoint(yaw_deg=yaw, pitch_deg=0.0, roll_deg=0.0)
            rendered_frames.append(frame)

        return {
            "mode": "3d",
            "scene": scene,
            "frames": rendered_frames
        }
