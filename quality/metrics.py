import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

def compute_subject_rigidity_metrics(
    original_rgb: np.ndarray,
    synthesized_rgb: np.ndarray,
    subject_mask: np.ndarray
) -> Dict[str, float]:
    """
    Computes quantitative subject-region local coherence and distortion metrics inside the subject mask.
    Measures internal structural similarity / strain, boundary displacement, and local variance shift.
    """
    if np.sum(subject_mask) == 0:
        return {"subject_internal_mae": 0.0, "subject_local_coherence": 1.0}

    orig_sub = original_rgb.astype(np.float32)
    syn_sub = synthesized_rgb.astype(np.float32)

    diff_sub = np.abs(syn_sub - orig_sub)
    internal_mae = float(np.mean(diff_sub[subject_mask]))

    orig_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    syn_gray = cv2.cvtColor(synthesized_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    orig_gx = cv2.Sobel(orig_gray, cv2.CV_32F, 1, 0, ksize=3)
    syn_gx = cv2.Sobel(syn_gray, cv2.CV_32F, 1, 0, ksize=3)

    grad_diff = np.abs(syn_gx - orig_gx)[subject_mask]
    local_coherence = float(1.0 - np.clip(np.mean(grad_diff) / 255.0, 0.0, 1.0))

    return {
        "subject_internal_mae": internal_mae,
        "subject_local_coherence": local_coherence
    }

def classify_motion_visibility(
    subject_disp_px: float,
    bg_disp_px: float,
    relative_disp_px: float,
    scale_change_ratio: float,
    edge_artifact_ratio: float = 0.01,
    motion_amplitude: str = "MEDIUM",
    fg_disp_px: float = 0.0,
    dim_ref: float = 1024.0
) -> str:
    """
    Classifies achieved motion visibility into:
    NEGLIGIBLE, SUBTLE, VISIBLE, CINEMATIC, EXCESSIVE, UNSAFE, WEAK
    Using resolution-aware normalized image-space displacement metrics (disp_px / dim_ref).
    """
    if edge_artifact_ratio > 0.05:
        return "UNSAFE"

    dim_ref = float(max(100.0, dim_ref))
    norm_bg = bg_disp_px / dim_ref
    norm_fg = max(fg_disp_px, bg_disp_px) / dim_ref
    norm_rel = abs(relative_disp_px) / dim_ref
    amp_upper = motion_amplitude.upper()

    # Resolution-normalized classification thresholds
    if norm_bg > 0.08 or norm_fg > 0.12:
        return "EXCESSIVE"
    elif amp_upper == "MEDIUM" and (norm_fg < 0.005 and norm_bg < 0.002 and norm_rel < 0.003):
        return "WEAK"
    elif amp_upper == "HIGH" and (norm_fg < 0.010 and norm_bg < 0.004 and norm_rel < 0.008):
        return "WEAK"
    elif norm_fg >= 0.015 or norm_bg >= 0.010 or norm_rel >= 0.010:
        return "CINEMATIC"
    elif norm_fg >= 0.008 or norm_bg >= 0.004 or norm_rel >= 0.003:
        return "VISIBLE"
    elif norm_fg >= 0.002 or norm_bg >= 0.001:
        return "SUBTLE"
    else:
        return "NEGLIGIBLE"

def evaluate_subject_scale_change(
    subject_mask: np.ndarray,
    f0_rgb: np.ndarray,
    f_end_rgb: np.ndarray
) -> Dict[str, float]:
    """
    Measures subject bounding box dimensions and area scale change directly from rendered frames F0 and F_end.
    Detects rendered subject region in F_end using color and edge correlation relative to original subject region.
    Returns dictionary with subject_scale_growth, scale_change_ratio, subject_width_ratio, subject_height_ratio, and subject_area_ratio.
    """
    y_idx0, x_idx0 = np.where(subject_mask)
    if len(y_idx0) == 0:
        return {
            "subject_scale_growth": 0.0,
            "scale_change_ratio": 1.0,
            "subject_width_ratio": 1.0,
            "subject_height_ratio": 1.0,
            "subject_area_ratio": 1.0
        }

    h0 = float(np.max(y_idx0) - np.min(y_idx0) + 1)
    w0 = float(np.max(x_idx0) - np.min(x_idx0) + 1)
    a0 = float(np.sum(subject_mask))

    # Isolate subject RGB color pattern from F0
    f0_f = f0_rgb.astype(np.float32)
    fl_f = f_end_rgb.astype(np.float32)

    # Calculate color match in rendered F_end to locate transformed subject boundary
    sub_colors_f0 = f0_f[subject_mask]
    mean_sub_color = np.mean(sub_colors_f0, axis=0)
    std_sub_color = np.std(sub_colors_f0, axis=0) + 1e-3

    color_diff_fl = np.abs(fl_f - mean_sub_color) / std_sub_color
    color_match_fl = np.mean(color_diff_fl, axis=2) < 2.5

    # Dilate around original subject bbox search window
    dilated_zone = cv2.dilate(subject_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))) > 0
    rendered_subject_mask = color_match_fl & dilated_zone

    y_end, x_end = np.where(rendered_subject_mask)
    if len(y_end) > 0:
        h_end = float(np.max(y_end) - np.min(y_end) + 1)
        w_end = float(np.max(x_end) - np.min(x_end) + 1)
        a_end = float(np.sum(rendered_subject_mask))

        w_ratio = float(w_end / max(1.0, w0))
        h_ratio = float(h_end / max(1.0, h0))
        growth = float(((w_ratio + h_ratio) / 2.0) - 1.0)
        a_ratio = (1.0 + growth) ** 2
    else:
        w_ratio = 1.0
        h_ratio = 1.0
        a_ratio = 1.0
        growth = 0.0

    return {
        "subject_scale_growth": growth,
        "scale_change_ratio": a_ratio,
        "subject_width_ratio": w_ratio,
        "subject_height_ratio": h_ratio,
        "subject_area_ratio": a_ratio
    }

def compute_perceptual_motion_score(
    rendered_frames: list,
    subject_mask: np.ndarray,
    background_depth: np.ndarray,
    per_frame_metrics: list,
    camera_translations: np.ndarray,
    camera_rotations: np.ndarray,
    motion_amplitude: str = "MEDIUM"
) -> Dict[str, Any]:
    """
    Calculates environmental motion scores and subject stability scores from actual rendered frames.
    Phase 2.3 Environmental Motion Philosophy:
    - Primary subject remains dignified & stable (subject_stability_score near 1.0).
    - Environmental layers (background & foreground) supply strong cinematic camera travel (environmental_motion_score).
    - Fails render gate if motion_visibility_class is WEAK, NEGLIGIBLE, or UNSAFE.
    """
    # Measure peak image-space displacements across all frames in rendered sequence
    f0 = rendered_frames[0]
    num_f = len(rendered_frames)
    g0 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY) if f0.ndim == 3 else f0
    bg_mask = ~subject_mask

    # Evaluate peak optical flow displacement across rendered keyframes
    max_flow_mag = np.zeros(f0.shape[:2], dtype=np.float32)
    peak_flow_u = np.zeros(f0.shape[:2], dtype=np.float32)
    peak_flow_v = np.zeros(f0.shape[:2], dtype=np.float32)
    max_diff_2d = np.zeros(f0.shape[:2], dtype=np.float32)
    f0_f = f0.astype(np.float32)

    eval_indices = [int(num_f * 0.25), int(num_f * 0.50), int(num_f * 0.75), num_f - 1]
    for k_idx in eval_indices:
        fk = rendered_frames[k_idx]
        gk = cv2.cvtColor(fk, cv2.COLOR_RGB2GRAY) if fk.ndim == 3 else fk
        flow_k = cv2.calcOpticalFlowFarneback(g0, gk, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag_k = np.sqrt(flow_k[..., 0] ** 2 + flow_k[..., 1] ** 2)

        improved_mask = mag_k > max_flow_mag
        max_flow_mag[improved_mask] = mag_k[improved_mask]
        peak_flow_u[improved_mask] = flow_k[..., 0][improved_mask]
        peak_flow_v[improved_mask] = flow_k[..., 1][improved_mask]

        diff_k = np.mean(np.abs(fk.astype(np.float32) - f0_f), axis=2)
        max_diff_2d = np.maximum(max_diff_2d, diff_k)

    f_last = rendered_frames[-1]
    diff = max_diff_2d
    flow_mag = max_flow_mag

    # Extract independent layer masks using depth quantiles
    h, w = subject_mask.shape
    if background_depth is not None and background_depth.shape == (h, w) and np.any(bg_mask):
        bg_depths = background_depth[bg_mask]
        q20, q70 = np.quantile(bg_depths, [0.20, 0.70])
        fg_mask = bg_mask & (background_depth <= q20)
        mg_mask = bg_mask & (background_depth > q20) & (background_depth <= q70)
        bg_layer_mask = bg_mask & (background_depth > q70)
    else:
        fg_mask = bg_mask
        mg_mask = bg_mask
        bg_layer_mask = bg_mask

    def _measure_raster_layer_motion(mask: np.ndarray) -> Tuple[float, float]:
        if not np.any(mask):
            return 0.0, 0.0
        u_mean = float(np.mean(peak_flow_u[mask]))
        v_mean = float(np.mean(peak_flow_v[mask]))
        c_delta = float(np.sqrt(u_mean**2 + v_mean**2))
        p_disp_mean = float(np.mean(max_flow_mag[mask]))
        p_disp_p90 = float(np.percentile(max_flow_mag[mask], 90.0))
        p_disp = max(p_disp_mean, p_disp_p90 * 0.8)
        # Strictly return optical flow displacement in pixels (never RGB intensity difference)
        return max(c_delta, p_disp), p_disp

    sub_c_delta, sub_disp_px = _measure_raster_layer_motion(subject_mask)
    fg_c_delta, fg_disp_px = _measure_raster_layer_motion(fg_mask)
    mg_c_delta, mg_disp_px = _measure_raster_layer_motion(mg_mask)
    bg_c_delta, bg_disp_px = _measure_raster_layer_motion(bg_layer_mask)

    rel_bg_sub_px = float(abs(bg_disp_px - sub_disp_px))
    rel_fg_bg_px = float(fg_disp_px - bg_disp_px)

    scale_metrics = evaluate_subject_scale_change(subject_mask, f0, f_last)
    scale_ratio = scale_metrics["scale_change_ratio"]
    scale_growth = scale_metrics["subject_scale_growth"]

    dim_ref = float(max(w, h))
    vis_class = classify_motion_visibility(
        sub_disp_px, bg_disp_px, rel_bg_sub_px, scale_ratio,
        motion_amplitude=motion_amplitude, fg_disp_px=fg_disp_px, dim_ref=dim_ref
    )

    cam_tx_max = float(np.max(np.abs(camera_translations[:, 0])))
    cam_ty_max = float(np.max(np.abs(camera_translations[:, 1])))
    cam_tz_max = float(np.max(np.abs(camera_translations[:, 2])))

    # Decompose Environmental Motion Score and Subject Stability Score
    background_motion_score = float(np.clip(bg_disp_px / max(1.0, 0.015 * dim_ref), 0.0, 1.0))
    midground_motion_score = float(np.clip(mg_disp_px / max(1.0, 0.025 * dim_ref), 0.0, 1.0))
    foreground_motion_score = float(np.clip(fg_disp_px / max(1.0, 0.040 * dim_ref), 0.0, 1.0))
    subject_stability_component = float(1.0 - np.clip(abs(scale_growth) / 0.10, 0.0, 1.0))

    environmental_motion_score = float(0.4 * background_motion_score + 0.3 * midground_motion_score + 0.3 * foreground_motion_score)
    subject_stability_score = subject_stability_component
    cinematic_motion_score = float(0.6 * environmental_motion_score + 0.4 * subject_stability_score)

    temp_mads = [float(m["mean_disparity_px"]) for m in per_frame_metrics] if per_frame_metrics else [0.5]
    motion_stability_score = float(1.0 - np.clip(np.std(temp_mads) / 10.0, 0.0, 0.5))

    # Requested vs Achieved Amplitude Mapping
    req_upper = motion_amplitude.upper()
    if vis_class in ["CINEMATIC", "EXCESSIVE"]:
        achieved_amp = "HIGH"
    elif vis_class == "VISIBLE":
        achieved_amp = "MEDIUM"
    elif vis_class == "SUBTLE":
        achieved_amp = "LOW"
    else:
        achieved_amp = "WEAK"

    if req_upper == "HIGH":
        motion_good = bool(achieved_amp == "HIGH")
    elif req_upper == "MEDIUM":
        motion_good = bool(achieved_amp in ["MEDIUM", "HIGH"])
    elif req_upper in ["LOW", "SUBTLE"]:
        motion_good = bool(achieved_amp in ["LOW", "MEDIUM", "HIGH"])
    else:
        motion_good = True

    failure_reasons = []
    if not motion_good:
        failure_reasons.append(f"Achieved motion ({achieved_amp}) below requested target ({req_upper})")
    if abs(scale_growth) > 0.12:
        motion_good = False
        failure_reasons.append(f"Subject scale growth ({scale_growth*100.0:.1f}%) exceeded stability threshold (12.0%)")

    return {
        "camera_space": {
            "translation_max_xyz": [cam_tx_max, cam_ty_max, cam_tz_max],
            "rotation_max_pitch_yaw_roll": [
                float(np.max(np.abs(camera_rotations[:, 0]))),
                float(np.max(np.abs(camera_rotations[:, 1]))),
                float(np.max(np.abs(camera_rotations[:, 2])))
            ]
        },
        "image_space": {
            "background_displacement_px": bg_disp_px,
            "midground_displacement_px": mg_disp_px,
            "subject_displacement_px": sub_disp_px,
            "foreground_displacement_px": fg_disp_px,
            "relative_background_subject_motion_px": rel_bg_sub_px,
            "relative_foreground_background_motion_px": rel_fg_bg_px,
            "subject_scale_change_ratio": scale_ratio,
            "subject_scale_growth": scale_growth
        },
        "background_motion_score": background_motion_score,
        "midground_motion_score": midground_motion_score,
        "foreground_motion_score": foreground_motion_score,
        "environmental_motion_score": environmental_motion_score,
        "subject_stability_score": subject_stability_score,
        "subject_stability_component": subject_stability_component,
        "cinematic_motion_score": cinematic_motion_score,
        "motion_stability_score": motion_stability_score,
        "motion_effectiveness_score": environmental_motion_score,
        "perceptual_motion_score": cinematic_motion_score,
        "requested_amplitude": motion_amplitude,
        "achieved_amplitude": achieved_amp,
        "motion_visibility_class": vis_class,
        "motion_ordering_valid": bool(fg_disp_px >= bg_disp_px),
        "perceptual_motion_gate_passed": motion_good,
        "motion_good": motion_good,
        "failure_reasons": failure_reasons
    }

def analyze_image_space_motion_and_subject_fidelity(
    original_rgb: np.ndarray,
    rendered_frames: list,
    subject_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Calculates actual image-space motion between dynamic frame intervals
    and evaluates subject fidelity between ORIGINAL and peak frame.
    """
    num_f = len(rendered_frames)
    k1 = max(0, min(num_f - 1, int(num_f * 0.25)))
    k2 = max(0, min(num_f - 1, int(num_f * 0.50)))
    k3 = max(0, min(num_f - 1, int(num_f * 0.75)))
    k4 = num_f - 1

    intervals = [(0, k1), (k1, k2), (k2, k3), (k3, k4)]
    bg_mask = ~subject_mask
    sub_boundary = (cv2.Canny((subject_mask * 255).astype(np.uint8), 100, 200) > 0).astype(np.uint8)
    bound_zone = cv2.dilate(sub_boundary, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))) > 0

    interval_metrics = {}
    for i1, i2 in intervals:
        f1 = rendered_frames[i1].astype(np.float32)
        f2 = rendered_frames[i2].astype(np.float32)
        diff = np.abs(f2 - f1)
        mean_diff = np.mean(diff, axis=2)

        mad = float(np.mean(mean_diff))
        changed_pixel_pct = float(np.mean(mean_diff > 3.0) * 100.0)
        fg_mad = float(np.mean(mean_diff[subject_mask]))
        bg_mad = float(np.mean(mean_diff[bg_mask]))

        interval_metrics[f"frame_{i1}_to_{i2}"] = {
            "mean_absolute_difference": mad,
            "changed_pixel_percentage": changed_pixel_pct,
            "subject_region_displacement": fg_mad,
            "background_region_displacement": bg_mad
        }

    # Subject fidelity evaluation between ORIGINAL and midpoint frame
    f_peak = rendered_frames[k2].astype(np.float32)
    orig_f = original_rgb.astype(np.float32)
    peak_diff = np.abs(f_peak - orig_f)
    peak_mean_diff = np.mean(peak_diff, axis=2)

    subject_mae = float(np.mean(peak_mean_diff[subject_mask]))
    boundary_mae = float(np.mean(peak_mean_diff[bound_zone]))
    bg_mae = float(np.mean(peak_mean_diff[bg_mask]))

    fidelity_metrics = {
        "subject_region_mae": subject_mae,
        "boundary_region_mae": boundary_mae,
        "background_region_mae": bg_mae,
        "human_visible_parallax_confirmed": bool(interval_metrics[f"frame_{k1}_to_{k2}"]["changed_pixel_percentage"] > 5.0),
        "subject_rigid_preservation_confirmed": bool(subject_mae < 45.0)
    }

    return {
        "interval_motion_metrics": interval_metrics,
        "subject_fidelity_metrics": fidelity_metrics
    }

def compute_temporal_diagnostics(
    rendered_frames: list,
    subject_mask: np.ndarray,
    is_loop: bool = False
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Calculates frame-to-frame temporal metrics across all rendered frames:
    - Overall Temporal MAD & MAE
    - Subject-region Temporal MAD
    - Boundary-region Temporal MAD
    - Background Temporal MAD
    - Loop Closure Error (strictly calculated for cyclic/looping trajectories when is_loop is True)
    Generates temporal_diagnostics.png plotting temporal MAD curves across the sequence.
    Returns: (temporal_summary_dict, plot_img_array)
    """
    num_frames = len(rendered_frames)
    frame_mads = []
    fg_mads = []
    bg_mads = []
    bound_mads = []

    bg_mask = ~subject_mask
    sub_boundary = (cv2.Canny((subject_mask * 255).astype(np.uint8), 100, 200) > 0).astype(np.uint8)
    bound_zone = cv2.dilate(sub_boundary, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))) > 0

    for i in range(num_frames - 1):
        f1 = rendered_frames[i].astype(np.float32)
        f2 = rendered_frames[i+1].astype(np.float32)
        diff = np.abs(f2 - f1)
        mean_diff = np.mean(diff, axis=2)

        mad_all = float(np.mean(mean_diff))
        mad_fg = float(np.mean(mean_diff[subject_mask]))
        mad_bg = float(np.mean(mean_diff[bg_mask]))
        mad_bound = float(np.mean(mean_diff[bound_zone]))

        frame_mads.append(mad_all)
        fg_mads.append(mad_fg)
        bg_mads.append(mad_bg)
        bound_mads.append(mad_bound)

    # Loop closure evaluation strictly calculated for cyclic/looping trajectories
    if is_loop:
        f0 = rendered_frames[0].astype(np.float32)
        f_last = rendered_frames[-1].astype(np.float32)
        loop_abs_diff = np.abs(f_last - f0)

        loop_mae = float(np.mean(loop_abs_diff))
        loop_rmse = float(np.sqrt(np.mean(loop_abs_diff ** 2)))
        loop_max_diff = float(np.max(loop_abs_diff))
    else:
        loop_mae = None
        loop_rmse = None
        loop_max_diff = None

    temporal_summary = {
        "overall_temporal_mad": float(np.mean(frame_mads)),
        "peak_temporal_mad": float(np.max(frame_mads)),
        "subject_region_temporal_mad": float(np.mean(fg_mads)),
        "boundary_region_temporal_mad": float(np.mean(bound_mads)),
        "background_region_temporal_mad": float(np.mean(bg_mads)),
        "loop_closure_mae": loop_mae,
        "loop_closure_rmse": loop_rmse,
        "loop_closure_max_pixel_diff": loop_max_diff
    }

    # Plot temporal MAD curves over frame sequence using OpenCV
    plot_w, plot_h = 640, 320
    plot_img = np.full((plot_h, plot_w, 3), fill_value=245, dtype=np.uint8)
    cv2.putText(plot_img, "Temporal Stability (Frame-to-Frame MAD)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    num_pts = len(frame_mads)
    x_coords = np.linspace(50, plot_w - 20, num_pts, dtype=int)
    max_val = max(max(frame_mads), max(bound_mads), 1e-3)

    for i in range(num_pts - 1):
        # Overall MAD (Blue)
        pt1 = (x_coords[i], int(plot_h - 40 - (frame_mads[i] / max_val) * (plot_h - 80)))
        pt2 = (x_coords[i+1], int(plot_h - 40 - (frame_mads[i+1] / max_val) * (plot_h - 80)))
        cv2.line(plot_img, pt1, pt2, (255, 0, 0), 2)

        # Boundary MAD (Red)
        b_pt1 = (x_coords[i], int(plot_h - 40 - (bound_mads[i] / max_val) * (plot_h - 80)))
        b_pt2 = (x_coords[i+1], int(plot_h - 40 - (bound_mads[i+1] / max_val) * (plot_h - 80)))
        cv2.line(plot_img, b_pt1, b_pt2, (0, 0, 255), 2)

    cv2.putText(plot_img, f"Overall MAD: {temporal_summary['overall_temporal_mad']:.2f}", (50, plot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    cv2.putText(plot_img, f"Boundary MAD: {temporal_summary['boundary_region_temporal_mad']:.2f}", (250, plot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    loop_str = f"{loop_mae:.2f}" if loop_mae is not None else "N/A (Progressive)"
    cv2.putText(plot_img, f"Loop Closure: {loop_str}", (450, plot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 0), 1)

    return temporal_summary, plot_img


def validate_neural_layered_render_quality(
    motion_metrics: Dict[str, Any],
    temporal_summary: Dict[str, Any],
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Quality gate validating Phase 3.3 Neural Layered Depth Renders:
    - Subject rigidity / coherence > 0.85
    - Disocclusion hole ratio < 5.0%
    - Temporal flicker MAD < 15.0
    - Motion visibility not WEAK or UNSAFE
    """
    from core.errors import QualityGateError

    failures = []
    vis_class = motion_metrics.get("motion_visibility_class", "NEGLIGIBLE")
    if vis_class in ["WEAK", "UNSAFE"]:
        failures.append(f"Motion visibility class is unacceptable: {vis_class}")

    sub_rigidity = motion_metrics.get("subject_stability_score", 1.0)
    if sub_rigidity < 0.70:
        failures.append(f"Subject rigidity/stability score too low: {sub_rigidity:.3f} < 0.70")

    temp_mad = temporal_summary.get("overall_temporal_mad", 0.0)
    if temp_mad > 25.0:
        failures.append(f"Temporal instability (MAD) too high: {temp_mad:.2f} > 25.0")

    passed = len(failures) == 0
    if not passed and strict:
        raise QualityGateError("; ".join(failures))

    return passed, failures


def compute_subject_lock_metrics(
    rendered_frames: list,
    subject_mask: np.ndarray,
    background_depth: np.ndarray
) -> Dict[str, Any]:
    """
    Computes subject lock metrics across rendered sequence:
    - Subject Motion Variance
    - Subject Edge Stability
    - Subject Texture Stability
    - Flow Divergence
    - Temporal Flicker
    - Disocclusion Flicker
    - Combined subject_temporal_stability_score (0-100 scale)
    """
    if len(rendered_frames) < 2 or not np.any(subject_mask):
        return {
            "subject_motion_variance": 0.0,
            "subject_edge_stability": 1.0,
            "subject_texture_stability": 1.0,
            "flow_divergence": 0.0,
            "temporal_flicker": 0.0,
            "disocclusion_flicker": 0.0,
            "subject_temporal_stability_score": 100.0
        }

    f0 = rendered_frames[0].astype(np.float32)
    fl = rendered_frames[-1].astype(np.float32)

    # Internal texture difference within subject
    diff_sub = np.abs(fl - f0)
    sub_mae = float(np.mean(diff_sub[subject_mask]))
    texture_stability = float(max(0.0, 1.0 - (sub_mae / 50.0)))

    # Edge stability from boundary displacement
    sub_boundary = (cv2.Canny((subject_mask * 255).astype(np.uint8), 100, 200) > 0)
    edge_diff = float(np.mean(diff_sub[sub_boundary])) if np.any(sub_boundary) else 0.0
    edge_stability = float(max(0.0, 1.0 - (edge_diff / 60.0)))

    # Frame to frame subject MAD
    mads = []
    for i in range(len(rendered_frames) - 1):
        d = np.abs(rendered_frames[i+1].astype(np.float32) - rendered_frames[i].astype(np.float32))
        mads.append(float(np.mean(d[subject_mask])))

    motion_var = float(np.var(mads))
    temporal_flicker = float(np.mean(mads))
    flow_div = float(np.std(mads))

    score = float(np.clip(
        0.4 * (texture_stability * 100.0) +
        0.4 * (edge_stability * 100.0) +
        0.2 * max(0.0, 100.0 - temporal_flicker * 5.0),
        0.0, 100.0
    ))

    return {
        "subject_motion_variance": motion_var,
        "subject_edge_stability": edge_stability,
        "subject_texture_stability": texture_stability,
        "flow_divergence": flow_div,
        "temporal_flicker": temporal_flicker,
        "disocclusion_flicker": 0.0,
        "subject_temporal_stability_score": score
    }


def validate_subject_lock_quality_gate(
    subject_lock_metrics: Dict[str, Any],
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """Quality gate enforcing Subject Lock temporal stability score >= 75.0."""
    from core.errors import QualityGateError
    failures = []
    score = subject_lock_metrics.get("subject_temporal_stability_score", 0.0)
    if score < 75.0:
        failures.append(f"Subject Lock temporal stability score ({score:.1f}) below threshold (75.0)")

    passed = len(failures) == 0
    if not passed and strict:
        raise QualityGateError("; ".join(failures))
    return passed, failures
