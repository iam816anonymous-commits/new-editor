"""
Inferred 3D Scene Builder.
"""

from scene_3d.scene import Inferred3DScene
from scene_3d.camera import PerspectiveCamera3D

class Inferred3DSceneBuilder:
    """Builder for constructing Inferred3DScene instances."""

    @staticmethod
    def build(rgb_array, depth_map, subject_mask):
        h, w, _ = rgb_array.shape
        camera = PerspectiveCamera3D(width=w, height=h, focal_length_x=float(w), focal_length_y=float(w), principal_point_x=w/2.0, principal_point_y=h/2.0)
        return Inferred3DScene(rgb_array=rgb_array, depth_map=depth_map, subject_mask=subject_mask, camera=camera)
