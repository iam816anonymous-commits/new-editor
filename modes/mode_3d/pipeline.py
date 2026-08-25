from typing import Dict, Any
import numpy as np
from core.contracts import RenderRequest
from scene_3d.scene import reconstruct_inferred_3d_scene, Inferred3DScene
from geometry.transforms import compute_rotation_matrix

class Mode3DPipeline:
    def __init__(self, request: RenderRequest):
        self.request = request

    def execute(self, rgb_array: np.ndarray, depth_map: np.ndarray, subject_mask: np.ndarray) -> Dict[str, Any]:
        scene = reconstruct_inferred_3d_scene(rgb_array=rgb_array, depth_map=depth_map, subject_mask=subject_mask)
        rendered_frames = []
        for yaw in np.linspace(-15.0, 15.0, num=self.request.frame_count):
            yaw_rad = np.radians(yaw)
            R_mat = compute_rotation_matrix(0.0, yaw_rad, 0.0)
            t_vec = np.zeros(3, dtype=np.float64)
            frame = scene.render_novel_view(R_mat, t_vec)
            rendered_frames.append(frame)
        return {"mode": "3d", "scene": scene, "frames": rendered_frames}
