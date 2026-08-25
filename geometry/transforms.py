import numpy as np

def compute_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Computes 3x3 rotation matrix from pitch, yaw, roll angles in radians."""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ], dtype=np.float64)

    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    return Rz @ Ry @ Rx

def transform_3d_points(points_3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Applies 3D rotation matrix R and translation vector t: P' = P @ R.T + t."""
    return (points_3d @ R.T) + t
