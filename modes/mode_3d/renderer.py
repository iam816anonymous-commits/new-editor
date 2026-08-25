"""
Inferred 3D Viewpoint Renderer.
"""

import numpy as np
from scene_3d.scene import Inferred3DScene

class Renderer3D:
    """Renders free-viewpoint sweeps over an Inferred3DScene."""

    @staticmethod
    def render(scene: Inferred3DScene, yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
        return scene.render_viewpoint(yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg)
