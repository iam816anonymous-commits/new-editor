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


def compute_normalized_depth(
    rendering_depth: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    gamma: float = 1.2
) -> np.ndarray:
    """
    Normalizes rendering depth Z in [min_depth, max_depth] into normalized depth coordinate space z_norm in [0, 1],
    applying depth gamma for nonlinear depth response.
    where 0.0 = far background and 1.0 = closest foreground.
    """
    span = max(1e-5, max_depth - min_depth)
    clipped = np.clip(rendering_depth, min_depth, max_depth)
    z_linear = 1.0 - (clipped - min_depth) / span
    z_gamma = np.power(np.clip(z_linear, 0.0, 1.0), gamma)
    return np.clip(z_gamma, 0.0, 1.0).astype(np.float32)


def compute_layer_motion_multiplier(layer_role_str: str, motion_amplitude: str = "MEDIUM") -> float:
    """
    Returns layer parallax motion multipliers for depth-aware 2.5D perspective projection.
    Strictly enforces layer parallax motion ordering: FOREGROUND > PRIMARY_SUBJECT > MIDGROUND > BACKGROUND.

    Presets:
    LOW:
      BACKGROUND: 0.25x, MIDGROUND: 0.50x, PRIMARY_SUBJECT: 0.85x, FOREGROUND: 1.25x
    MEDIUM:
      BACKGROUND: 0.45x, MIDGROUND: 0.90x, PRIMARY_SUBJECT: 1.60x, FOREGROUND: 2.50x
    HIGH:
      BACKGROUND: 0.75x, MIDGROUND: 1.50x, PRIMARY_SUBJECT: 2.60x, FOREGROUND: 4.20x
    """
    role_upper = layer_role_str.upper()
    amp_upper = motion_amplitude.upper()

    if amp_upper == "LOW":
        multipliers = {
            "BACKGROUND": 0.25,
            "MIDGROUND": 0.50,
            "PRIMARY_SUBJECT": 0.85,
            "PRIMARY_SUBJECT_PART": 0.85,
            "FOREGROUND": 1.25,
            "ANALYSIS_ONLY": 0.10
        }
    elif amp_upper == "HIGH":
        multipliers = {
            "BACKGROUND": 0.75,
            "MIDGROUND": 1.50,
            "PRIMARY_SUBJECT": 2.60,
            "PRIMARY_SUBJECT_PART": 2.60,
            "FOREGROUND": 4.20,
            "ANALYSIS_ONLY": 0.20
        }
    else:  # MEDIUM default
        multipliers = {
            "BACKGROUND": 0.45,
            "MIDGROUND": 0.90,
            "PRIMARY_SUBJECT": 1.60,
            "PRIMARY_SUBJECT_PART": 1.60,
            "FOREGROUND": 2.50,
            "ANALYSIS_ONLY": 0.15
        }

    return multipliers.get(role_upper, 0.45)


def compute_layer_disparity(
    camera: CameraModel,
    tx: float,
    depth_fg: float,
    depth_sub: float,
    depth_bg: float
) -> Dict[str, float]:
    """
    Computes theoretical 2D screen disparity in pixels across scene depth layers
    for camera horizontal translation tx using bounded layer motion multipliers.
    Guarantees Displacement_fg > Displacement_sub > Displacement_bg when Z_fg < Z_sub < Z_bg.
    """
    mult_fg = compute_layer_motion_multiplier("FOREGROUND")
    mult_sub = compute_layer_motion_multiplier("PRIMARY_SUBJECT")
    mult_bg = compute_layer_motion_multiplier("BACKGROUND")

    disp_fg = (abs(camera.fx * tx) / max(0.01, depth_fg)) * mult_fg
    disp_sub = (abs(camera.fx * tx) / max(0.01, depth_sub)) * mult_sub
    disp_bg = (abs(camera.fx * tx) / max(0.01, depth_bg)) * mult_bg

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
    """Exports camera intrinsics and generalized frame trajectory poses to JSON-serializable dictionary."""
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
