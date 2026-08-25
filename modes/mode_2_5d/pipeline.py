"""
Mode A (2.5D Parallax) Pipeline Implementation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from core.contracts import RenderRequest
from rendering.frame_renderer import verify_zero_motion_identity
from rendering.sequence_renderer import render_full_frame_sequence

class Mode25DPipeline:
    """Standalone 2.5D Parallax Pipeline Execution Engine."""

    def __init__(self, request: RenderRequest):
        self.request = request

    def execute(
        self,
        rgb_array: np.ndarray,
        depth_map: np.ndarray,
        subject_mask: np.ndarray,
        bg_plate: np.ndarray,
        bg_depth: np.ndarray,
        provenance_map: Optional[np.ndarray] = None,
        boundary_risk_map: Optional[np.ndarray] = None,
        translations: Optional[np.ndarray] = None,
        rotations: Optional[np.ndarray] = None,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: float = 250.0,
        cy: float = 250.0,
        disparity_ceiling_px: float = 50.0,
        frames_dir: Optional[Path] = None,
        spatial_diagnostics: Optional[Any] = None,
        motion_amplitude: str = "MEDIUM"
    ) -> Dict[str, Any]:
        """Executes 2.5D view synthesis sequence rendering."""
        if provenance_map is None:
            provenance_map = np.ones_like(depth_map, dtype=np.float32)
        if boundary_risk_map is None:
            boundary_risk_map = np.zeros_like(depth_map, dtype=np.float32)

        identity_pass = verify_zero_motion_identity(
            rgb_array, depth_map, bg_plate, bg_depth, provenance_map,
            fx, fy, cx, cy
        )

        frame_count = self.request.frame_count if self.request else 48
        if translations is None:
            translations = np.zeros((frame_count, 3), dtype=np.float64)
        if rotations is None:
            rotations = np.zeros((frame_count, 3), dtype=np.float64)
        if frames_dir is None:
            frames_dir = Path("output/frames")

        frames, per_frame_metrics = render_full_frame_sequence(
            rgb_array=rgb_array,
            depth_map=depth_map,
            bg_plate=bg_plate,
            bg_depth=bg_depth,
            provenance_map=provenance_map,
            subject_mask=subject_mask,
            boundary_risk_map=boundary_risk_map,
            translations=translations,
            rotations=rotations,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            disparity_ceiling_px=disparity_ceiling_px,
            frames_dir=frames_dir,
            spatial_diagnostics=spatial_diagnostics,
            motion_amplitude=motion_amplitude,
            frame_count=frame_count
        )

        return {
            "mode": "2.5d",
            "frames": frames,
            "per_frame_metrics": per_frame_metrics,
            "zero_motion_identity_passed": identity_pass
        }
