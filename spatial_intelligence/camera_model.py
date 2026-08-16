"""
Camera Model Abstraction for 2.5D Perspective Projection and Parallax Calculation.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
from .schemas import CameraModel


def create_perspective_camera(
    width: int,
    height: int,
    fov_degrees: float = 53.0
) -> CameraModel:
    """Creates a perspective pinhole camera model derived from image dimensions."""
    focal_length = float(max(width, height))
    cx = width / 2.0
    cy = height / 2.0
    return CameraModel(
        width=width,
        height=height,
        fx=focal_length,
        fy=focal_length,
        cx=cx,
        cy=cy
    )


def compute_layer_disparity(
    camera: CameraModel,
    tx: float,
    depth_fg: float,
    depth_sub: float,
    depth_bg: float
) -> Dict[str, float]:
    """
    Computes theoretical 2D screen disparity in pixels across scene depth layers
    for camera horizontal translation tx.
    Guarantees Displacement_fg > Displacement_sub > Displacement_bg when Z_fg < Z_sub < Z_bg.
    """
    disp_fg = abs(camera.fx * tx) / max(0.01, depth_fg)
    disp_sub = abs(camera.fx * tx) / max(0.01, depth_sub)
    disp_bg = abs(camera.fx * tx) / max(0.01, depth_bg)

    return {
        "tx_camera_units": tx,
        "foreground_disparity_px": float(disp_fg),
        "subject_disparity_px": float(disp_sub),
        "background_disparity_px": float(disp_bg),
        "relative_parallax_sub_vs_bg_px": float(disp_sub - disp_bg)
    }


def export_camera_path_dict(
    camera: CameraModel,
    translations: np.ndarray,
    rotations: np.ndarray
) -> Dict[str, Any]:
    """Exports camera intrinsics and 48-frame trajectory poses to JSON-serializable dictionary."""
    frames_path = []
    num_frames = len(translations)

    for i in range(num_frames):
        frames_path.append({
            "frame_index": i,
            "translation": [round(float(c), 6) for c in translations[i]],
            "rotation_pitch_yaw_roll_rad": [round(float(r), 6) for r in rotations[i]]
        })

    return {
        "intrinsics": {
            "width": camera.width,
            "height": camera.height,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy
        },
        "frame_count": num_frames,
        "camera_trajectory": frames_path
    }
