import numpy as np
from typing import Optional, Tuple

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


def regularize_subject_depth(
    depth_map: np.ndarray,
    subject_mask: np.ndarray,
    guide_image: Optional[np.ndarray] = None,
    spatial_sigma: float = 5.0,
    range_sigma: float = 0.05
) -> np.ndarray:
    """
    Regularizes monocular depth in the subject region to suppress high-frequency depth noise,
    preventing independent pixel swimming while preserving edge-aware 3D geometric structure.
    Uses joint bilateral filtering and median-guided blending inside the subject mask.
    """
    import cv2
    subj_bool = subject_mask.astype(bool)
    if not np.any(subj_bool):
        return depth_map.copy()

    reg_depth = depth_map.copy().astype(np.float32)
    d_min, d_max = float(np.min(depth_map[subj_bool])), float(np.max(depth_map[subj_bool]))
    d_span = max(d_max - d_min, 1e-5)

    # 8-bit normalized depth inside subject
    d_norm = np.clip((depth_map - d_min) / d_span * 255.0, 0, 255).astype(np.uint8)

    # Edge-aware bilateral filter on depth
    filtered_norm = cv2.bilateralFilter(
        d_norm, d=7, sigmaColor=range_sigma * 255.0, sigmaSpace=spatial_sigma
    )

    filtered_depth = d_min + (filtered_norm.astype(np.float32) / 255.0) * d_span

    # Smooth blending towards local median depth near soft boundary
    subj_median = float(np.median(depth_map[subj_bool]))
    mask_uint8 = (subj_bool * 255).astype(np.uint8)
    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    max_dist = np.max(dist) if np.max(dist) > 0 else 1.0
    core_weight = np.clip(dist / (0.3 * max_dist), 0.0, 1.0)

    # Blend: 80% regularized structural depth + 20% global median, weighted by distance from boundary
    blended_subj_depth = core_weight * filtered_depth + (1.0 - core_weight) * subj_median
    reg_depth[subj_bool] = blended_subj_depth[subj_bool]

    return reg_depth


def compute_subject_rigid_transform(
    R_cam: np.ndarray,
    t_cam: np.ndarray,
    subject_depth: float,
    subject_motion_scale: float = 0.75
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a temporally smooth SE(3) rigid transformation (R_subj, t_subj)
    for the primary subject derived from the main camera trajectory.
    Uncouples internal texture rigidity from global camera movement:
    - Z-translation (scale growth for push-in/dolly) uses full camera translation (1.0x).
    - Lateral XY translations use subject_motion_scale (0.75x) to maintain stable focal anchor while enabling strong relative parallax.
    """
    t_subj = t_cam.copy()
    # Apply 0.75x scale for lateral XY moves while preserving 1.0x Z camera movement for perspective scale change
    t_subj[0] *= float(subject_motion_scale)
    t_subj[1] *= float(subject_motion_scale)
    # Full Z translation for scale expansion
    t_subj[2] *= 1.0

    # Smooth rotation coupling (75% of camera rotation)
    R_subj = np.eye(3, dtype=np.float64) + 0.75 * (R_cam - np.eye(3, dtype=np.float64))

    return R_subj, t_subj
