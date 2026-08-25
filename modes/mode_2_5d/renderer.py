"""
2.5D View Synthesis Renderer.
"""

import numpy as np
from geometry.splatting import render_single_frame_forward_splatting

class Renderer25D:
    """2.5D View Synthesis Engine performing Z-buffered forward subpixel splatting."""

    @staticmethod
    def render_frame(
        rgb_array: np.ndarray,
        depth_map: np.ndarray,
        translation: np.ndarray,
        rotation: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float
    ) -> np.ndarray:
        return render_single_frame_forward_splatting(
            rgb_array, depth_map, translation, rotation, fx, fy, cx, cy
        )
