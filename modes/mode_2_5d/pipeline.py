"""
Mode A (2.5D Parallax) Pipeline Implementation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from core.contracts import RenderRequest
from camera.intrinsics import derive_camera_intrinsics
from camera.safety import plan_safe_motion_trajectory
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
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        disparity_ceiling_px: Optional[float] = None,
        frames_dir: Optional[Path] = None,
        spatial_diagnostics: Optional[Any] = None,
        motion_amplitude: str = "MEDIUM"
    ) -> Dict[str, Any]:
        """Executes 2.5D view synthesis sequence rendering using real camera intrinsics and trajectories."""
        h, w = rgb_array.shape[:2]

        if fx is None or fy is None or cx is None or cy is None:
            fx, fy, cx, cy = derive_camera_intrinsics(w, h)

        if provenance_map is None:
            provenance_map = np.ones_like(depth_map, dtype=np.float32)
        if boundary_risk_map is None:
            boundary_risk_map = np.zeros_like(depth_map, dtype=np.float32)

        frame_count = self.request.frame_count if self.request else 48
        motion_style = self.request.motion_style if self.request else "Cinematic Push-In"
        motion_strength = self.request.motion_strength if self.request else "Cinematic"

        if translations is None or rotations is None or disparity_ceiling_px is None:
            confidence_map = np.ones_like(depth_map, dtype=np.float32)
            trans, rots, scale, plan_sum = plan_safe_motion_trajectory(
                style=motion_style,
                strength=motion_strength,
                width=w,
                height=h,
                depth_map=depth_map,
                confidence_map=confidence_map,
                subject_mask=subject_mask,
                boundary_risk_map=boundary_risk_map,
                provenance_map=provenance_map,
                fx=fx, fy=fy, cx=cx, cy=cy,
                num_frames=frame_count
            )
            if translations is None:
                translations = trans
            if rotations is None:
                rotations = rots
            if disparity_ceiling_px is None:
                disparity_ceiling_px = plan_sum["disparity_ceiling_target_px"]

        identity_pass = verify_zero_motion_identity(
            rgb_array, depth_map, bg_plate, bg_depth, provenance_map,
            fx, fy, cx, cy
        )

        if frames_dir is None:
            out_base = self.request.output_dir if self.request else Path("output")
            frames_dir = out_base / "frames"

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
