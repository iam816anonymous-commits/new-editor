import numpy as np
from typing import Tuple

def back_project_points(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> np.ndarray:
    """
    Back-projects 2D image coordinates (u, v) and continuous depth Z to 3D points [X, Y, Z].

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = Z

    Returns array of shape (N, 3).
    """
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.column_stack([X, Y, Z])

def project_3d_points(
    points_3d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects 3D points [X, Y, Z] to image coordinates (u', v') and depth Z'.

    u' = fx * X / Z + cx
    v' = fy * Y / Z + cy
    """
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    # Avoid division by zero
    Z_safe = np.where(np.abs(Z) < 1e-6, 1e-6, Z)

    u_proj = fx * (X / Z_safe) + cx
    v_proj = fy * (Y / Z_safe) + cy
    return u_proj, v_proj, Z
