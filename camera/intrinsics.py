import numpy as np
from typing import Tuple

def derive_camera_intrinsics(width: int, height: int) -> Tuple[float, float, float, float]:
    """
    Derives rendering camera intrinsics (fx, fy, cx, cy) from image dimensions.
    Assumes standard pinhole perspective field of view (~53 degrees vertical FOV).
    """
    focal_length = float(max(width, height))
    cx = width / 2.0
    cy = height / 2.0
    return focal_length, focal_length, cx, cy


def compute_adaptive_convergence_depth(
    depth_map: np.ndarray,
    subject_mask: float | np.ndarray | None = None
) -> float:
    """
    Computes adaptive convergence depth (zero-parallax plane) based on subject depth median
    or robust scene depth median.
    """
    if subject_mask is not None and np.any(subject_mask):
        subj_bool = subject_mask.astype(bool)
        return float(np.median(depth_map[subj_bool]))
    else:
        return float(np.median(depth_map))
