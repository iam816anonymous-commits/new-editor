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
