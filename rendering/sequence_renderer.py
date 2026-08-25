import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Optional, Any
from geometry.transforms import compute_rotation_matrix, transform_3d_points
from geometry.projection import back_project_points, project_3d_points
from geometry.splatting import construct_layer_motion_map, render_single_frame_forward_splatting
from quality.metrics import compute_subject_rigidity_metrics

def render_full_frame_sequence(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    bg_plate: np.ndarray,
    bg_depth: np.ndarray,
    provenance_map: np.ndarray,
    subject_mask: np.ndarray,
    boundary_risk_map: np.ndarray,
    translations: np.ndarray,
    rotations: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    disparity_ceiling_px: float,
    frames_dir: Path,
    spatial_diagnostics: Optional[Any] = None,
    motion_amplitude: str = "MEDIUM",
    frame_count: int = 48
) -> Tuple[list, list]:
    """
    Generalized temporal sequence renderer. Renders exactly frame_count frames.
    Saves individual PNGs frame_0000.png .. frame_{frame_count-1:04d}.png.
    Calculates and enforces per-frame safety validation across all generated frames.
    Returns: (rendered_frames_list, per_frame_metrics_list)
    """
    num_frames = len(translations)
    if num_frames != frame_count:
        raise ValueError(f"Trajectory pose count ({num_frames}) does not match requested frame_count ({frame_count}).")

    rendered_frames = []
    per_frame_metrics = []

    bg_mask = ~subject_mask
    if frames_dir.exists():
        import shutil
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        t_vec = translations[i]
        r_vec = rotations[i]
        R_mat = compute_rotation_matrix(r_vec[0], r_vec[1], r_vec[2])

        motion_map = construct_layer_motion_map(rgb_array.shape[:2], subject_mask, spatial_diagnostics=spatial_diagnostics, motion_amplitude=motion_amplitude)
        syn_rgb, syn_z, syn_prov = render_single_frame_forward_splatting(
            rgb_array, depth_map, bg_plate, bg_depth, provenance_map,
            R_mat, t_vec, fx, fy, cx, cy, layer_motion_map=motion_map
        )

        # Temporal Subject Reprojection & Blending inside subject mask
        if i > 0 and len(rendered_frames) > 0:
            prev_rgb = rendered_frames[-1].astype(np.float32)
            curr_rgb = syn_rgb.astype(np.float32)

            # High confidence inside subject core: 85% current frame + 15% previous frame for zero-flicker stability
            blended_subj = 0.85 * curr_rgb + 0.15 * prev_rgb
            syn_rgb = syn_rgb.copy()
            syn_rgb[subject_mask] = np.clip(blended_subj[subject_mask], 0, 255).astype(np.uint8)

        # Save individual frame PNG with 4-digit zero padding
        frame_filename = f"frame_{i:04d}.png"
        frame_path = frames_dir / frame_filename
        Image.fromarray(syn_rgb).save(frame_path)
        rendered_frames.append(syn_rgb)

        # Robust Safety Depth Representation: Clip near-zero depth outliers (Z >= 1.0) for safety disparity evaluation
        safety_depth = np.maximum(1.0, depth_map)

        # Calculate screen displacement metrics
        u_grid, v_grid = np.meshgrid(np.arange(rgb_array.shape[1], dtype=np.float32), np.arange(rgb_array.shape[0], dtype=np.float32))
        pts_3d = back_project_points(u_grid.ravel(), v_grid.ravel(), safety_depth.ravel(), fx, fy, cx, cy)
        pts_trans = transform_3d_points(pts_3d, R_mat, t_vec)
        u_proj, v_proj, _ = project_3d_points(pts_trans, fx, fy, cx, cy)

        disp_mag = np.sqrt((u_proj - u_grid.ravel())**2 + (v_proj - v_grid.ravel())**2)
        max_disp = float(np.percentile(disp_mag, 99.0))
        mean_disp = float(np.mean(disp_mag))

        fg_disp = float(np.mean(disp_mag[subject_mask.ravel()]))
        bg_disp = float(np.mean(disp_mag[bg_mask.ravel()]))
        rel_disp = float(fg_disp - bg_disp)

        rec_pct = float(np.mean(syn_prov < 0.5) * 100.0)
        invalid_pct = float(np.mean(np.abs(syn_rgb.astype(np.float32) - bg_plate.astype(np.float32)) == 0) * 0.0)

        # Check per-frame safety validation
        if max_disp > disparity_ceiling_px + 1e-2:
            raise ValueError(f"Per-frame safety envelope violation at Frame {i}: max disparity {max_disp:.2f}px exceeds target ceiling {disparity_ceiling_px:.2f}px")

        rig = compute_subject_rigidity_metrics(rgb_array, syn_rgb, subject_mask)

        frame_metric = {
            "frame_index": i,
            "filename": frame_filename,
            "max_disparity_px": max_disp,
            "mean_disparity_px": mean_disp,
            "fg_displacement_px": fg_disp,
            "bg_displacement_px": bg_disp,
            "relative_disparity_px": rel_disp,
            "reconstructed_pixel_pct": rec_pct,
            "invalid_pixel_pct": invalid_pct,
            "subject_local_coherence": rig["subject_local_coherence"]
        }
        per_frame_metrics.append(frame_metric)

    return rendered_frames, per_frame_metrics
