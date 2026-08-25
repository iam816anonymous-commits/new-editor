"""
Mode A (2.5D Parallax) Pipeline Implementation.
"""

from typing import Dict, Any
import numpy as np
from core.contracts import RenderRequest, RenderResult
from rendering.frame_renderer import verify_zero_motion_identity
from rendering.sequence_renderer import render_full_frame_sequence

class Mode25DPipeline:
    """Standalone 2.5D Parallax Pipeline Execution Engine."""

    def __init__(self, request: RenderRequest):
        self.request = request

    def execute(self, rgb_array: np.ndarray, depth_map: np.ndarray, subject_mask: np.ndarray, bg_plate: np.ndarray, bg_depth: np.ndarray) -> Dict[str, Any]:
        """Executes 2.5D view synthesis sequence rendering."""
        # Zero motion identity verification
        identity_pass = verify_zero_motion_identity(rgb_array, depth_map)

        # Sequence rendering
        frames = render_full_frame_sequence(
            rgb_array=rgb_array,
            depth_map=depth_map,
            subject_mask=subject_mask,
            bg_plate=bg_plate,
            bg_depth=bg_depth,
            motion_style=self.request.motion_style,
            motion_strength=self.request.motion_strength,
            frame_count=self.request.frame_count
        )
        return {
            "mode": "2.5d",
            "frames": frames,
            "zero_motion_identity_passed": identity_pass
        }
