import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from geometry.projection import back_project_points, project_3d_points
from geometry.transforms import compute_rotation_matrix, transform_3d_points
from geometry.splatting import construct_layer_motion_map, render_single_frame_forward_splatting
from quality.metrics import compute_subject_rigidity_metrics

def verify_zero_motion_identity(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    bg_plate: np.ndarray,
    bg_depth: np.ndarray,
    provenance_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Renders with zero motion (R = Identity, t = 0) to verify identity reprojection.
    Computes quantitative error metrics: MAE, RMSE, Max Absolute Pixel Error, and Percentage Differing Pixels (>2 L1 diff).
    """
    R_identity = np.eye(3, dtype=np.float64)
    t_zero = np.zeros(3, dtype=np.float64)

    syn_rgb, _, _ = render_single_frame_forward_splatting(
        rgb_array, depth_map, bg_plate, bg_depth, provenance_map,
        R_identity, t_zero, fx, fy, cx, cy
    )

    # Absolute difference
    abs_diff = np.abs(syn_rgb.astype(np.float32) - rgb_array.astype(np.float32))
    diff_vis = np.clip(np.mean(abs_diff, axis=2) * 10.0, 0, 255).astype(np.uint8)  # 10x boosted visualization

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(abs_diff ** 2)))
    max_err = float(np.max(abs_diff))
    differing_pixel_pct = float(np.mean(abs_diff > 2.0) * 100.0)

    metrics = {
        "zero_motion_mae": mae,
        "zero_motion_rmse": rmse,
        "zero_motion_max_pixel_error": max_err,
        "zero_motion_differing_pixel_pct": differing_pixel_pct
    }

    return syn_rgb, diff_vis, metrics

def run_micro_motion_sweep(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    bg_plate: np.ndarray,
    bg_depth: np.ndarray,
    provenance_map: np.ndarray,
    subject_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    tx_fractions: Optional[list] = None
) -> Tuple[Dict[float, np.ndarray], Dict[float, Dict[str, float]]]:
    """
    Executes a true micro-motion sweep over small controlled translations (e.g. 0.0025, 0.005, 0.01, 0.015 of scene width).
    Calculates exact screen-space displacements in PIXELS for foreground, background, relative disparity,
    and reconstructed/invalid pixel percentages.
    """
    if tx_fractions is None:
        tx_fractions = [0.0025, 0.005, 0.01, 0.015]

    width = rgb_array.shape[1]
    R_identity = np.eye(3, dtype=np.float64)

    sweep_frames = {}
    sweep_metrics = {}

    bg_mask = ~subject_mask

    for frac in tx_fractions:
        # Camera displacement in scene units relative to width
        t_x = frac * width / fx  # camera horizontal shift
        t_vec = np.array([t_x, 0.0, 0.0], dtype=np.float64)

        syn_rgb, syn_z, syn_prov = render_single_frame_forward_splatting(
            rgb_array, depth_map, bg_plate, bg_depth, provenance_map,
            R_identity, t_vec, fx, fy, cx, cy
        )

        sweep_frames[frac] = syn_rgb

        fg_depths = depth_map[subject_mask]
        bg_depths = depth_map[bg_mask]

        fg_disparity_px = (fx * t_x) / fg_depths
        bg_disparity_px = (fx * t_x) / bg_depths

        mean_fg_disp = float(np.mean(fg_disparity_px))
        mean_bg_disp = float(np.mean(bg_disparity_px))
        relative_disparity = float(mean_fg_disp - mean_bg_disp)
        max_disparity = float(np.max(fg_disparity_px))
        mean_disparity = float(np.mean(np.concatenate([fg_disparity_px, bg_disparity_px])))

        rec_pct = float(np.mean(syn_prov < 0.5) * 100.0)

        abs_diff = np.abs(syn_rgb.astype(np.float32) - rgb_array.astype(np.float32))
        diff_pct = float(np.mean(abs_diff > 5.0) * 100.0)

        sweep_metrics[frac] = {
            "fraction_width": frac,
            "tx_camera_units": t_x,
            "fg_displacement_px": mean_fg_disp,
            "bg_displacement_px": mean_bg_disp,
            "relative_disparity_px": relative_disparity,
            "max_disparity_px": max_disparity,
            "mean_disparity_px": mean_disparity,
            "reconstructed_pixel_pct": rec_pct,
            "differing_pixel_pct": diff_pct
        }

    return sweep_frames, sweep_metrics

def synthesize_micro_motion_frame(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    bg_plate: np.ndarray,
    bg_depth: np.ndarray,
    provenance_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    t_x: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Renders one single frame with a small horizontal camera translation t_x.
    Calculates displacement and exposure metrics.
    """
    R_identity = np.eye(3, dtype=np.float64)
    t_vec = np.array([t_x, 0.0, 0.0], dtype=np.float64)

    micro_rgb, micro_z, micro_prov = render_single_frame_forward_splatting(
        rgb_array, depth_map, bg_plate, bg_depth, provenance_map,
        R_identity, t_vec, fx, fy, cx, cy
    )

    abs_diff = np.abs(micro_rgb.astype(np.float32) - rgb_array.astype(np.float32))
    diff_vis = np.clip(np.mean(abs_diff, axis=2) * 5.0, 0, 255).astype(np.uint8)

    subpixel_diff_pct = float(np.mean(abs_diff > 5.0) * 100.0)
    reconstructed_pixel_pct = float(np.mean(micro_prov < 0.5) * 100.0)

    metrics = {
        "micro_motion_tx": t_x,
        "micro_motion_differing_pixel_pct": subpixel_diff_pct,
        "micro_motion_reconstructed_pixel_pct": reconstructed_pixel_pct
    }

    return micro_rgb, diff_vis, micro_prov, metrics

def render_phase_e_representative_keyframes(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    bg_plate: np.ndarray,
    bg_depth: np.ndarray,
    provenance_map: np.ndarray,
    subject_mask: np.ndarray,
    translations: np.ndarray,
    rotations: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    spatial_diagnostics: Optional[Any] = None,
    motion_amplitude: str = "MEDIUM"
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """
    Renders 5 representative keyframes along the planned trajectory:
    start (0%), 25%, 50%, 75%, and end (100%).
    Calculates detailed frame metrics: FG displacement, BG displacement, relative disparity,
    reconstructed pixel %, and subject rigidity.
    """
    num_frames = len(translations)
    indices = {
        "start": 0,
        "25": num_frames // 4,
        "50": num_frames // 2,
        "75": (num_frames * 3) // 4,
        "end": num_frames - 1
    }

    keyframes = {}
    keyframe_metrics = {}
    bg_mask = ~subject_mask

    for name, idx in indices.items():
        t_vec = translations[idx]
        r_vec = rotations[idx]
        R_mat = compute_rotation_matrix(r_vec[0], r_vec[1], r_vec[2])

        amp = getattr(args, "motion_amplitude", "MEDIUM") if 'args' in locals() else "MEDIUM"
        motion_map = construct_layer_motion_map(rgb_array.shape[:2], subject_mask, spatial_diagnostics=spatial_diagnostics, motion_amplitude=motion_amplitude)
        syn_rgb, syn_z, syn_prov = render_single_frame_forward_splatting(
            rgb_array, depth_map, bg_plate, bg_depth, provenance_map,
            R_mat, t_vec, fx, fy, cx, cy, layer_motion_map=motion_map
        )
        keyframes[name] = syn_rgb

        # Calculate screen-space displacements in pixels
        u_grid, v_grid = np.meshgrid(np.arange(rgb_array.shape[1], dtype=np.float32), np.arange(rgb_array.shape[0], dtype=np.float32))
        pts_3d = back_project_points(u_grid.ravel(), v_grid.ravel(), depth_map.ravel(), fx, fy, cx, cy)
        pts_trans = transform_3d_points(pts_3d, R_mat, t_vec)
        u_proj, v_proj, _ = project_3d_points(pts_trans, fx, fy, cx, cy)

        disp_x = np.abs(u_proj - u_grid.ravel())
        disp_y = np.abs(v_proj - v_grid.ravel())
        disp_mag = np.sqrt(disp_x**2 + disp_y**2)

        fg_disp = float(np.mean(disp_mag[subject_mask.ravel()]))
        bg_disp = float(np.mean(disp_mag[bg_mask.ravel()]))
        rel_disp = float(fg_disp - bg_disp)

        rig = compute_subject_rigidity_metrics(rgb_array, syn_rgb, subject_mask)
        rec_pct = float(np.mean(syn_prov < 0.5) * 100.0)

        keyframe_metrics[name] = {
            "frame_index": idx,
            "fg_displacement_px": fg_disp,
            "bg_displacement_px": bg_disp,
            "relative_disparity_px": rel_disp,
            "max_disparity_px": float(np.percentile(disp_mag, 99.0)),
            "reconstructed_pixel_pct": rec_pct,
            "subject_internal_mae": rig["subject_internal_mae"],
            "subject_local_coherence": rig["subject_local_coherence"]
        }

    return keyframes, keyframe_metrics
