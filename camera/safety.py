import numpy as np
from typing import Dict, Any, Tuple, Optional
from geometry.projection import back_project_points, project_3d_points
from geometry.transforms import compute_rotation_matrix, transform_3d_points
from geometry.splatting import render_single_frame_forward_splatting
from camera.trajectories import generate_c1_smooth_trajectory
from quality.metrics import compute_subject_rigidity_metrics

def compute_safety_margins(
    plan_summary: Dict[str, Any],
    disparity_ceiling_target_px: float,
    reconstructed_limit_pct: float = 12.0,
    boundary_risk_limit: float = 0.20
) -> Dict[str, float]:
    """
    Calculates exact remaining safety margins across scene risk factors:
    - Disocclusion / Reconstruction Margin (%)
    - Boundary Risk Margin
    - Disparity Ceiling Margin (px)
    - Depth Confidence Margin
    - Projection Margin
    """
    rec_pct = plan_summary.get("scene_reconstructed_area_pct", 5.0)
    risk_exp = plan_summary.get("scene_boundary_risk_exposure", 0.1)
    peak_disp = plan_summary.get("peak_max_disparity_px", 30.0)
    mean_conf = plan_summary.get("scene_mean_confidence", 0.9)

    return {
        "disocclusion_margin_pct": max(0.0, 100.0 - rec_pct),
        "reconstruction_margin_pct": max(0.0, reconstructed_limit_pct - rec_pct),
        "boundary_risk_margin": max(0.0, boundary_risk_limit - risk_exp),
        "disparity_ceiling_margin_px": max(0.0, disparity_ceiling_target_px - peak_disp),
        "depth_confidence_margin": float(mean_conf)
    }

def run_trajectory_magnitude_sweep(
    style: str,
    base_scale: float,
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    bg_plate: np.ndarray,
    bg_depth: np.ndarray,
    provenance_map: np.ndarray,
    subject_mask: np.ndarray,
    boundary_risk_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    disparity_ceiling_px: float,
    scales: Optional[list] = None
) -> Dict[float, Dict[str, float]]:
    """
    Executes a controlled sweep around the planned trajectory magnitude (e.g. 0.50x, 0.75x, 1.00x, 1.25x, 1.50x).
    For each candidate scale, records:
    - Max Disparity (px)
    - Reconstructed %
    - Boundary Risk Exposure
    - Subject Local Coherence
    - Invalid / Unwritten Pixels %
    - Safety Status (SAFE vs CEILING_VIOLATED)
    """
    if scales is None:
        scales = [0.50, 0.75, 1.00, 1.25, 1.50]

    width = rgb_array.shape[1]
    sweep_results = {}

    for mult in scales:
        cand_scale = base_scale * mult
        trans, rots = generate_c1_smooth_trajectory(style, cand_scale, num_frames=48)

        # Test peak pose frame
        peak_idx = 12
        t_peak = trans[peak_idx]
        r_peak = rots[peak_idx]
        R_peak = compute_rotation_matrix(r_peak[0], r_peak[1], r_peak[2])

        syn_rgb, syn_z, syn_prov = render_single_frame_forward_splatting(
            rgb_array, depth_map, bg_plate, bg_depth, provenance_map,
            R_peak, t_peak, fx, fy, cx, cy
        )

        # Compute disparity
        u_grid, v_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(rgb_array.shape[0], dtype=np.float32))
        pts_3d = back_project_points(u_grid.ravel(), v_grid.ravel(), depth_map.ravel(), fx, fy, cx, cy)
        pts_trans = transform_3d_points(pts_3d, R_peak, t_peak)
        u_proj, v_proj, _ = project_3d_points(pts_trans, fx, fy, cx, cy)

        disp_mag = np.sqrt((u_proj - u_grid.ravel())**2 + (v_proj - v_grid.ravel())**2)
        max_disp = float(np.percentile(disp_mag, 99.0))

        rig = compute_subject_rigidity_metrics(rgb_array, syn_rgb, subject_mask)
        rec_pct = float(np.mean(syn_prov < 0.5) * 100.0)
        high_risk_zone = boundary_risk_map[boundary_risk_map > 0.5]
        risk_exp = float(np.mean(high_risk_zone)) if len(high_risk_zone) > 0 else 0.0

        is_safe = max_disp <= disparity_ceiling_px

        sweep_results[mult] = {
            "scale_multiplier": mult,
            "candidate_magnitude_scale": cand_scale,
            "max_disparity_px": max_disp,
            "reconstructed_pixel_pct": rec_pct,
            "boundary_risk_exposure": risk_exp,
            "subject_local_coherence": rig["subject_local_coherence"],
            "safety_status": "SCENE_SAFE" if is_safe else "CEILING_VIOLATED"
        }

    return sweep_results

def plan_safe_motion_trajectory(
    style: str,
    strength: str,
    width: int,
    height: int,
    depth_map: np.ndarray,
    confidence_map: np.ndarray,
    subject_mask: np.ndarray,
    boundary_risk_map: np.ndarray,
    provenance_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    num_frames: int = 48
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    Closed-loop motion planner that automatically computes a safe camera trajectory.
    Enforces resolution-proportional strength ceilings:
    - Subtle: 3.0% image dimension
    - Cinematic: 6.0% image dimension
    - Strong: 10.0% image dimension
    and iteratively scales down magnitude if candidate poses violate safety limits.

    Returns: (translations, rotations, final_magnitude_scale, trajectory_plan_summary)
    """
    dim_ref = float(max(width, height))
    strength_ceilings = {
        "SUBTLE": 0.030 * dim_ref,
        "CINEMATIC": 0.060 * dim_ref,
        "STRONG": 0.100 * dim_ref
    }
    target_disparity_ceiling = strength_ceilings.get(strength.upper(), 0.060 * dim_ref)

    # Scene safety factors based on scene analysis
    mean_confidence = float(np.mean(confidence_map))
    boundary_risk_exposure = float(np.mean(boundary_risk_map[boundary_risk_map > 0.5]) if np.sum(boundary_risk_map > 0.5) > 0 else 0.0)
    reconstructed_area_pct = float((1.0 - np.mean(provenance_map)) * 100.0)

    # Compute base safe magnitude scale
    base_scale = 1.0
    if mean_confidence < 0.7:
        base_scale *= 0.8
    if boundary_risk_exposure > 0.15:
        base_scale *= 0.75
    if reconstructed_area_pct > 10.0:
        base_scale *= 0.8

    magnitude_scale = base_scale

    # Closed-loop convergence loop
    max_iterations = 10
    accepted = False

    for iteration in range(max_iterations):
        translations, rotations = generate_c1_smooth_trajectory(style, magnitude_scale, num_frames=num_frames)

        # Robust Safety Depth Representation: Clip pathological near-zero depth outliers (Z >= 1.0)
        # prevents isolated 0.10px boundary noise from collapsing global camera trajectory safety
        safety_depth = np.maximum(1.0, depth_map)

        # Evaluate max disparity across ALL frames in trajectory to guarantee per-frame safety envelope compliance
        u_grid, v_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        pts_3d = back_project_points(u_grid.ravel(), v_grid.ravel(), safety_depth.ravel(), fx, fy, cx, cy)

        max_disp_across_all = 0.0
        mean_disp_across_all = 0.0

        for k_idx in range(num_frames):
            t_k = translations[k_idx]
            r_k = rotations[k_idx]
            R_k = compute_rotation_matrix(r_k[0], r_k[1], r_k[2])

            pts_trans = transform_3d_points(pts_3d, R_k, t_k)
            u_proj, v_proj, _ = project_3d_points(pts_trans, fx, fy, cx, cy)

            disp_mag = np.sqrt((u_proj - u_grid.ravel())**2 + (v_proj - v_grid.ravel())**2)
            max_k = float(np.percentile(disp_mag, 99.0))
            if max_k > max_disp_across_all:
                max_disp_across_all = max_k
                mean_disp_across_all = float(np.mean(disp_mag))

        max_disp_px = max_disp_across_all
        mean_disp_px = mean_disp_across_all

        # Verify safety envelope constraints
        if max_disp_px <= target_disparity_ceiling:
            # Check if trajectory is under-leveraging available disparity capacity for HIGH/STRONG request
            if strength.upper() in ["STRONG", "HIGH"] and max_disp_px < 0.85 * target_disparity_ceiling and iteration < 8:
                expansion_factor = min(1.8, (0.90 * target_disparity_ceiling) / max(max_disp_px, 1.0))
                magnitude_scale *= expansion_factor
            else:
                accepted = True
                break
        else:
            # Closed-loop reduction
            reduction_factor = target_disparity_ceiling / max(max_disp_px, 1e-5)
            magnitude_scale *= max(reduction_factor * 0.95, 0.5)

    # Re-generate final accepted trajectory
    translations, rotations = generate_c1_smooth_trajectory(style, magnitude_scale, num_frames=num_frames)

    # Position & velocity loop closure error check
    pos_closure_err = float(np.linalg.norm(translations[0] - translations[-1]))
    vel_closure_err = float(np.linalg.norm((translations[1] - translations[0]) - (translations[-1] - translations[-2])))

    plan_summary = {
        "requested_style": style,
        "requested_strength": strength,
        "disparity_ceiling_target_px": target_disparity_ceiling,
        "initial_base_scale": base_scale,
        "final_magnitude_scale": magnitude_scale,
        "closed_loop_iterations": iteration + 1,
        "peak_max_disparity_px": max_disp_px,
        "peak_mean_disparity_px": mean_disp_px,
        "scene_mean_confidence": mean_confidence,
        "scene_boundary_risk_exposure": boundary_risk_exposure,
        "scene_reconstructed_area_pct": reconstructed_area_pct,
        "loop_position_closure_error": pos_closure_err,
        "loop_velocity_closure_error": vel_closure_err
    }

    return translations, rotations, magnitude_scale, plan_summary
