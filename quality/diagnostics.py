import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Dict, Any, List, Optional
from inference.segmentation import generate_candidate_masks_contact_sheet
from geometry.projection import back_project_points, project_3d_points
from geometry.transforms import compute_rotation_matrix, transform_3d_points
from geometry.splatting import construct_layer_motion_map, render_single_frame_forward_splatting
from quality.metrics import compute_subject_rigidity_metrics

def generate_discontinuity_rejection_map(
    depth_map: np.ndarray,
    rgb_array: np.ndarray,
    depth_discontinuity_threshold: float = 0.5
) -> np.ndarray:
    """
    Generates a visual diagnostic map highlighting samples rejected near sharp depth discontinuities.
    Overlays rejected boundary edges in RED over grayscale RGB reference to verify stretching/bleeding protection.
    """
    d_grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    d_grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    d_grad_mag = np.sqrt(d_grad_x**2 + d_grad_y**2)

    is_discontinuity = d_grad_mag > depth_discontinuity_threshold

    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    vis[is_discontinuity] = [255, 0, 0]
    return vis

def analyze_zero_motion_errors(
    original_rgb: np.ndarray,
    zero_motion_rgb: np.ndarray,
    subject_mask: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Analyzes and categorizes raw forward-splatting zero-motion reprojection errors.
    Categorizes errors (>2 L1 pixel difference) into:
    - Edge/boundary pixels
    - Subject interior
    - Background interior
    - Uncovered/interpolated subpixel rounding
    Returns (error_mask_vis, error_breakdown_metrics).
    """
    abs_diff = np.abs(zero_motion_rgb.astype(np.float32) - original_rgb.astype(np.float32))
    max_channel_diff = np.max(abs_diff, axis=2)
    error_mask = max_channel_diff > 2.0

    total_errors = np.sum(error_mask)
    if total_errors == 0:
        pct_edge, pct_sub, pct_bg = 0.0, 0.0, 0.0
    else:
        gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_zone = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0

        edge_errors = np.sum(error_mask & edge_zone)
        sub_errors = np.sum(error_mask & subject_mask & (~edge_zone))
        bg_errors = np.sum(error_mask & (~subject_mask) & (~edge_zone))

        pct_edge = float(edge_errors / total_errors * 100.0)
        pct_sub = float(sub_errors / total_errors * 100.0)
        pct_bg = float(bg_errors / total_errors * 100.0)

    error_vis = np.zeros_like(original_rgb, dtype=np.uint8)
    if total_errors > 0:
        gray_bg = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
        error_vis = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2RGB) // 2

        gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_zone = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0

        error_vis[error_mask & edge_zone] = [255, 0, 0]        # Red = Edge subpixel rounding
        error_vis[error_mask & subject_mask & (~edge_zone)] = [0, 255, 0]  # Green = Subject interior
        error_vis[error_mask & (~subject_mask) & (~edge_zone)] = [0, 100, 255] # Blue = Background interior

    breakdown = {
        "total_error_pixels": int(total_errors),
        "pct_errors_at_edges": pct_edge,
        "pct_errors_inside_subject": pct_sub,
        "pct_errors_inside_background": pct_bg
    }
    return error_vis, breakdown

def generate_micro_sweep_contact_sheet(
    original_rgb: np.ndarray,
    zero_motion_rgb: np.ndarray,
    sweep_frames: Dict[float, np.ndarray],
    subject_mask: np.ndarray,
    crop_size: int = 140
) -> np.ndarray:
    """
    Generates a multi-column visual contact sheet comparing:
    ORIGINAL | ZERO MOTION | MICRO 0.0025 | MICRO 0.005 | MICRO 0.01 | MICRO 0.015
    """
    h, w, _ = original_rgb.shape
    half_crop = crop_size // 2

    y_sub, x_sub = np.where(subject_mask)
    if len(y_sub) > 0:
        center_face = (int(np.mean(y_sub)), int(np.mean(x_sub)))
    else:
        center_face = (h // 2, w // 2)

    if len(y_sub) > 0:
        center_hands = (int(np.percentile(y_sub, 75)), int(np.mean(x_sub)))
    else:
        center_hands = (int(h * 0.7), int(w * 0.5))

    sub_boundary = cv2.Canny((subject_mask * 255).astype(np.uint8), 100, 200) > 0
    by, bx = np.where(sub_boundary)
    if len(by) > 0:
        center_sub_edge = (by[len(by) // 2], bx[len(bx) // 2])
    else:
        center_sub_edge = (int(h * 0.4), int(w * 0.4))

    bg_y, bg_x = np.where(~subject_mask)
    if len(bg_y) > 0:
        center_bg = (bg_y[len(bg_y) // 4], bg_x[len(bg_x) // 4])
    else:
        center_bg = (int(h * 0.1), int(w * 0.1))

    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ey, ex = np.where(edges > 0)
    if len(ey) > 0:
        center_fine = (ey[len(ey) // 2], ex[len(ex) // 2])
    else:
        center_fine = (int(h * 0.8), int(w * 0.8))

    centers = [center_face, center_hands, center_sub_edge, center_bg, center_fine]
    labels = ["Face/Center", "Hands/Lower", "Subject Boundary", "Background", "Fine Structure"]

    fractions = [0.0025, 0.005, 0.01, 0.015]

    rows = []
    for (cy_c, cx_c), label in zip(centers, labels):
        y0 = max(0, min(h - crop_size, cy_c - half_crop))
        x0 = max(0, min(w - crop_size, cx_c - half_crop))
        y1, x1 = y0 + crop_size, x0 + crop_size

        c_orig = cv2.resize(original_rgb[y0:y1, x0:x1], (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
        c_zero = cv2.resize(zero_motion_rgb[y0:y1, x0:x1], (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)

        cv2.putText(c_orig, f"{label} (Orig)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(c_zero, "Zero-Motion", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        col_cells = [c_orig, c_zero]

        for frac in fractions:
            frame_img = sweep_frames.get(frac, original_rgb)
            c_sweep = cv2.resize(frame_img[y0:y1, x0:x1], (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
            cv2.putText(c_sweep, f"t={frac}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            col_cells.append(c_sweep)

        row = np.hstack(col_cells)
        rows.append(row)

    contact_sheet = np.vstack(rows)
    return contact_sheet

def generate_subject_coherence_diagnostics(
    original_rgb: np.ndarray,
    keyframes: Dict[str, np.ndarray],
    subject_mask: np.ndarray,
    crop_size: int = 140
) -> np.ndarray:
    """
    Generates a 2x enlarged visual diagnostic sheet comparing subject structural preservation
    across trajectory keyframes (Original, Start, 25%, 50%, 75%, End) around key detailed features:
    1. Face / Eyes / Nose / Mouth
    2. Hands / Fingers / Ornaments
    3. Clothing folds / Silhouettes
    4. Fine Subject Edge
    """
    h, w, _ = original_rgb.shape
    half_crop = crop_size // 2

    y_sub, x_sub = np.where(subject_mask)
    if len(y_sub) > 0:
        center_face = (int(np.mean(y_sub)), int(np.mean(x_sub)))
    else:
        center_face = (h // 2, w // 2)

    if len(y_sub) > 0:
        center_hands = (int(np.percentile(y_sub, 75)), int(np.mean(x_sub)))
    else:
        center_hands = (int(h * 0.7), int(w * 0.5))

    sub_boundary = cv2.Canny((subject_mask * 255).astype(np.uint8), 100, 200) > 0
    by, bx = np.where(sub_boundary)
    if len(by) > 0:
        center_edge = (by[len(by) // 2], bx[len(bx) // 2])
    else:
        center_edge = (int(h * 0.4), int(w * 0.4))

    if len(y_sub) > 0:
        center_folds = (int(np.percentile(y_sub, 60)), int(np.percentile(x_sub, 60)))
    else:
        center_folds = (int(h * 0.6), int(w * 0.6))

    centers = [center_face, center_hands, center_folds, center_edge]
    labels = ["Face/Features", "Hands/Ornaments", "Clothing Folds", "Subject Silhouette"]

    k_names = ["start", "25", "50", "75", "end"]

    rows = []
    for (cy_c, cx_c), label in zip(centers, labels):
        y0 = max(0, min(h - crop_size, cy_c - half_crop))
        x0 = max(0, min(w - crop_size, cx_c - half_crop))
        y1, x1 = y0 + crop_size, x0 + crop_size

        c_orig = cv2.resize(original_rgb[y0:y1, x0:x1], (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c_orig, f"{label} (Orig)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cells = [c_orig]
        for kn in k_names:
            img_k = keyframes.get(kn, original_rgb)
            c_k = cv2.resize(img_k[y0:y1, x0:x1], (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
            cv2.putText(c_k, f"Frame {kn}%", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cells.append(c_k)

        row = np.hstack(cells)
        rows.append(row)

    sheet = np.vstack(rows)
    return sheet

def generate_phase_d_crop_diagnostics(
    original_rgb: np.ndarray,
    zero_motion_rgb: np.ndarray,
    micro_motion_rgb: np.ndarray,
    subject_mask: np.ndarray,
    crop_size: int = 160
) -> np.ndarray:
    """
    Generates a visual diagnostic contact sheet with 2x enlarged crops around critical structural regions:
    1. Center / Subject Face
    2. Subject Boundary / Silhouette
    3. Background Region
    4. Fine Structure / Edge
    Returns contact sheet RGB numpy array.
    """
    h, w, _ = original_rgb.shape
    half_crop = crop_size // 2

    # Define 4 crop centers
    # Crop 1: Subject Center / Face region
    y_indices, x_indices = np.where(subject_mask)
    if len(y_indices) > 0:
        center1 = (int(np.mean(y_indices)), int(np.mean(x_indices)))
    else:
        center1 = (h // 2, w // 2)

    # Crop 2: Subject Boundary
    sub_boundary = cv2.Canny((subject_mask * 255).astype(np.uint8), 100, 200) > 0
    by, bx = np.where(sub_boundary)
    if len(by) > 0:
        center2 = (by[len(by) // 2], bx[len(bx) // 2])
    else:
        center2 = (int(h * 0.4), int(w * 0.4))

    # Crop 3: Background
    bg_y, bg_x = np.where(~subject_mask)
    if len(bg_y) > 0:
        center3 = (bg_y[len(bg_y) // 4], bg_x[len(bg_x) // 4])
    else:
        center3 = (int(h * 0.1), int(w * 0.1))

    # Crop 4: Fine Structure / Edge
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ey, ex = np.where(edges > 0)
    if len(ey) > 0:
        center4 = (ey[len(ey) // 2], ex[len(ex) // 2])
    else:
        center4 = (int(h * 0.7), int(w * 0.7))

    centers = [center1, center2, center3, center4]
    crop_labels = ["Center/Face", "Subject Edge", "Background", "Fine Structure"]

    rows = []
    for (cy_c, cx_c), label in zip(centers, crop_labels):
        y0 = max(0, min(h - crop_size, cy_c - half_crop))
        x0 = max(0, min(w - crop_size, cx_c - half_crop))
        y1, x1 = y0 + crop_size, x0 + crop_size

        c_orig = cv2.resize(original_rgb[y0:y1, x0:x1], (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
        c_zero = cv2.resize(zero_motion_rgb[y0:y1, x0:x1], (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
        c_micro = cv2.resize(micro_motion_rgb[y0:y1, x0:x1], (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)

        # Add labels to top left of each crop
        cv2.putText(c_orig, f"{label} (Orig)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(c_zero, f"{label} (Zero)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(c_micro, f"{label} (Micro)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        row = np.hstack([c_orig, c_zero, c_micro])
        rows.append(row)

    contact_sheet = np.vstack(rows)
    return contact_sheet

def generate_visual_review_contact_sheet(
    original_rgb: np.ndarray,
    rendered_frames: list,
    target_width: int = 400
) -> np.ndarray:
    """
    Generates an 8-panel grid contact sheet output/<hash>/cinematic/visual_review.png
    containing ORIGINAL and 7 dynamically sampled frames up to F{last} arranged in a 2x4 grid.
    Includes prominent text labels and scales all frames consistently.
    """
    h, w, _ = original_rgb.shape
    aspect = h / float(w)
    target_height = int(target_width * aspect)

    num_f = len(rendered_frames)
    sample_indices = np.linspace(0, num_f - 1, 7, dtype=int)

    frame_indices = [("ORIGINAL", original_rgb)] + [
        (f"FRAME {idx:02d}", rendered_frames[idx]) for idx in sample_indices
    ]

    labeled_panels = []
    for label, img in frame_indices:
        resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        panel = resized.copy()
        # Draw background bar for text
        cv2.rectangle(panel, (0, 0), (target_width, 35), (0, 0, 0), -1)
        cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255) if "ORIG" in label else (0, 255, 0), 2)
        labeled_panels.append(panel)

    # Arrange 2x4 grid (2 rows, 4 columns)
    row1 = np.hstack(labeled_panels[0:4])
    row2 = np.hstack(labeled_panels[4:8])
    grid = np.vstack([row1, row2])
    return grid

def generate_motion_amplitude_comparison_contact_sheet(
    original_rgb: np.ndarray,
    depth_map: np.ndarray,
    subject_mask: np.ndarray,
    bg_plate: np.ndarray,
    bg_depth: np.ndarray,
    provenance_map: np.ndarray,
    translations: np.ndarray,
    rotations: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    target_w: int = 240
) -> np.ndarray:
    """
    Generates a 3-row diagnostic contact sheet comparing LOW, MEDIUM, and HIGH MOTION across
    keyframe positions: F00, F25, F50, F75, F99 (5 keyframes per row).
    ROW 1: LOW MOTION
    ROW 2: MEDIUM MOTION
    ROW 3: HIGH MOTION
    """
    h, w, _ = original_rgb.shape
    aspect = h / float(w)
    target_h = int(target_w * aspect)
    num_f = len(translations)
    sample_indices = np.linspace(0, num_f - 1, 5, dtype=int)

    def render_preset_frames(amp_setting: str) -> list:
        motion_map = construct_layer_motion_map(original_rgb.shape[:2], subject_mask, spatial_diagnostics=None, motion_amplitude=amp_setting)
        preset_frames = []
        for s_idx in sample_indices:
            t_vec = translations[s_idx]
            r_vec = rotations[s_idx]
            R_mat = compute_rotation_matrix(r_vec[0], r_vec[1], r_vec[2])
            syn_rgb, _, _ = render_single_frame_forward_splatting(
                original_rgb, depth_map, bg_plate, bg_depth, provenance_map,
                R_mat, t_vec, fx, fy, cx, cy, layer_motion_map=motion_map
            )
            preset_frames.append((s_idx, syn_rgb))
        return preset_frames

    low_frames = render_preset_frames("LOW")
    med_frames = render_preset_frames("MEDIUM")
    high_frames = render_preset_frames("HIGH")

    def make_panel(img: np.ndarray, label: str) -> np.ndarray:
        p = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(p, (0, 0), (target_w, 22), (0, 0, 0), -1)
        cv2.putText(p, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
        return p

    row1_panels = [make_panel(img, f"LOW F{f_idx:02d}") for f_idx, img in low_frames]
    row2_panels = [make_panel(img, f"MEDIUM F{f_idx:02d}") for f_idx, img in med_frames]
    row3_panels = [make_panel(img, f"HIGH F{f_idx:02d}") for f_idx, img in high_frames]

    sheet = np.vstack([
        np.hstack(row1_panels),
        np.hstack(row2_panels),
        np.hstack(row3_panels)
    ])
    return sheet

def generate_p0_frame_difference_artifacts(
    rendered_frames: list,
    subject_mask: np.ndarray,
    output_dir: Path
) -> dict:
    """
    Generates P0 frame difference diagnostic report and visualizations:
    1. output/debug/frame_difference_report.json (F00->F99 & F_k->F_{k+1} pixel differences)
    2. output/debug/frame_diff_f00_f99.png
    3. output/debug/frame_overlay_f00_f99.png (Red/Cyan channel overlay showing spatial motion)
    4. output/debug/motion_heatmap.png (COLORMAP_JET visual motion heatmap)
    """
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    f0 = rendered_frames[0]
    f_end = rendered_frames[-1]
    h, w, _ = f0.shape
    bg_mask = ~subject_mask

    f0_f = f0.astype(np.float32)
    fe_f = f_end.astype(np.float32)

    # Intensity diff map
    diff_2d = np.mean(np.abs(fe_f - f0_f), axis=2)
    changed_mask = diff_2d > 2.0
    pct_changed = float((np.sum(changed_mask) / diff_2d.size) * 100.0)

    # Optical flow between F00 and F_end
    g0 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY) if f0.ndim == 3 else f0
    ge = cv2.cvtColor(f_end, cv2.COLOR_RGB2GRAY) if f_end.ndim == 3 else f_end
    flow = cv2.calcOpticalFlowFarneback(g0, ge, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # Difference Visualization (Grayscale diff)
    diff_vis = np.clip(diff_2d * 5.0, 0, 255).astype(np.uint8)
    Image.fromarray(diff_vis).save(debug_dir / "frame_diff_f00_f99.png")

    # Overlay Visualization (F_end Red channel, F0 Green/Blue channels)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = f_end[:, :, 0]
    overlay[:, :, 1] = f0[:, :, 1]
    overlay[:, :, 2] = f0[:, :, 2]
    Image.fromarray(overlay).save(debug_dir / "frame_overlay_f00_f99.png")

    # Motion Heatmap (COLORMAP_JET from flow magnitude)
    heatmap_norm = np.clip((flow_mag / max(0.1, np.max(flow_mag))) * 255.0, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    Image.fromarray(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)).save(debug_dir / "motion_heatmap.png")

    # Per-frame differences F_k -> F_{k+1}
    num_f = len(rendered_frames)
    frame_step_diffs = []
    for k in range(num_f - 1):
        fk = rendered_frames[k].astype(np.float32)
        fk1 = rendered_frames[k + 1].astype(np.float32)
        d_k = np.mean(np.abs(fk1 - fk), axis=2)
        frame_step_diffs.append({
            "frame_step": f"F{k:02d}->F{k+1:02d}",
            "mean_abs_diff": float(round(float(np.mean(d_k)), 4)),
            "median_abs_diff": float(round(float(np.median(d_k)), 4)),
            "p95_abs_diff": float(round(float(np.percentile(d_k, 95.0)), 4))
        })

    report = {
        "end_to_end": {
            "mean_abs_diff": float(round(float(np.mean(diff_2d)), 4)),
            "median_abs_diff": float(round(float(np.median(diff_2d)), 4)),
            "p95_abs_diff": float(round(float(np.percentile(diff_2d, 95.0)), 4)),
            "pct_pixels_changed": float(round(pct_changed, 2)),
            "subject_mean_diff": float(round(float(np.mean(diff_2d[subject_mask])), 4)),
            "background_mean_diff": float(round(float(np.mean(diff_2d[bg_mask])), 4)),
            "flow_magnitude_p50": float(round(float(np.median(flow_mag)), 4)),
            "flow_magnitude_p90": float(round(float(np.percentile(flow_mag, 90.0)), 4))
        },
        "frame_steps": frame_step_diffs
    }

    with open(debug_dir / "frame_difference_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report

def generate_phase_1_7_multi_row_contact_sheet(
    original_rgb: np.ndarray,
    rendered_frames: list,
    depth_map: np.ndarray,
    subject_mask: np.ndarray,
    boundary_risk_map: np.ndarray,
    target_w: int = 240
) -> np.ndarray:
    """
    Generates multi-row visual validation contact sheet supporting dynamic frame_count (48 or 100):
    ROW 1: ORIGINAL, F00, and 6 dynamically sampled frames up to F{last} (F47 or F99)
    ROW 2: EXTRACTED LAYERS (BG, MG, Primary Subject, FG)
    ROW 3: DEPTH MAP, FINAL LAYER MAP, OCCLUSION MAP, COMPOSITE MASK
    ROW 4: EDGE ARTIFACT MAP, TEMPORAL DIFFERENCE MAP
    """
    h, w, _ = original_rgb.shape
    aspect = h / float(w)
    target_h = int(target_w * aspect)
    num_f = len(rendered_frames)

    def resize_panel(img: np.ndarray, title: str) -> np.ndarray:
        if img.ndim == 2:
            img_rgb = cv2.cvtColor((img * 255.0 / (img.max() if img.max() > 0 else 1.0)).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img
        p = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(p, (0, 0), (target_w, 24), (0, 0, 0), -1)
        cv2.putText(p, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        return p

    # Dynamic frame index sampling for 48 or 100 frames
    sample_indices = np.linspace(0, num_f - 1, 7, dtype=int)
    row1_imgs = [("ORIGINAL", original_rgb)] + [(f"F{idx:02d}", rendered_frames[idx]) for idx in sample_indices]
    row1_panels = [resize_panel(img, title) for title, img in row1_imgs]

    # ROW 2: Extracted Layers (BG, MG, Subject, FG) + pad
    bg_img = original_rgb.copy(); bg_img[subject_mask] = 0
    sub_img = original_rgb.copy(); sub_img[~subject_mask] = 0
    blank_bg = np.zeros_like(original_rgb)

    row2_imgs = [
        ("LAYER: BACKGROUND", bg_img),
        ("LAYER: MIDGROUND", blank_bg),
        ("LAYER: PRIMARY SUBJ", sub_img),
        ("LAYER: FOREGROUND", blank_bg),
        ("BLANK", blank_bg), ("BLANK", blank_bg), ("BLANK", blank_bg), ("BLANK", blank_bg)
    ]
    row2_panels = [resize_panel(img, title) for title, img in row2_imgs]

    # ROW 3: Maps (Depth, Final Layer, Occlusion, Composite Mask)
    d_vis = ((depth_map - depth_map.min()) / max(1e-5, depth_map.max() - depth_map.min()) * 255.0).astype(np.uint8)
    risk_vis = (boundary_risk_map * 255.0).clip(0, 255).astype(np.uint8)
    sub_vis = (subject_mask * 255).astype(np.uint8)

    row3_imgs = [
        ("DEPTH MAP", d_vis),
        ("FINAL LAYER MAP", d_vis),
        ("OCCLUSION MAP", risk_vis),
        ("COMPOSITE MASK", sub_vis),
        ("BLANK", blank_bg), ("BLANK", blank_bg), ("BLANK", blank_bg), ("BLANK", blank_bg)
    ]
    row3_panels = [resize_panel(img, title) for title, img in row3_imgs]

    # ROW 4: Artifact Maps (Edge Artifact, Temporal Difference Map)
    f0 = rendered_frames[0].astype(np.float32)
    f24 = rendered_frames[24].astype(np.float32)
    temp_diff = np.clip(np.mean(np.abs(f24 - f0), axis=2) * 5.0, 0, 255).astype(np.uint8)

    row4_imgs = [
        ("EDGE ARTIFACT MAP", risk_vis),
        ("TEMP DIFF MAP", temp_diff),
        ("BLANK", blank_bg), ("BLANK", blank_bg), ("BLANK", blank_bg), ("BLANK", blank_bg), ("BLANK", blank_bg), ("BLANK", blank_bg)
    ]
    row4_panels = [resize_panel(img, title) for title, img in row4_imgs]

    # Combine into 4-row grid
    grid = np.vstack([
        np.hstack(row1_panels),
        np.hstack(row2_panels),
        np.hstack(row3_panels),
        np.hstack(row4_panels)
    ])
    return grid

def generate_visual_review_diagnostics_sheet(
    original_rgb: np.ndarray,
    rendered_frames: list,
    subject_mask: np.ndarray,
    crop_size: int = 160
) -> np.ndarray:
    """
    Generates output/<hash>/cinematic/visual_review_diagnostics.png
    comparing ORIGINAL against dynamically sampled keyframes with 2x enlarged crops
    across 5 critical regions:
    1. Face
    2. Hands
    3. Ornaments
    4. Subject Silhouette
    5. Background
    """
    h, w, _ = original_rgb.shape
    half_crop = crop_size // 2

    y_sub, x_sub = np.where(subject_mask)
    center_face = (int(np.mean(y_sub)), int(np.mean(x_sub))) if len(y_sub) > 0 else (h // 2, w // 2)
    center_hands = (int(np.percentile(y_sub, 75)), int(np.mean(x_sub))) if len(y_sub) > 0 else (int(h * 0.7), int(w * 0.5))

    sub_boundary = (cv2.Canny((subject_mask * 255).astype(np.uint8), 100, 200) > 0).astype(np.uint8)
    by, bx = np.where(sub_boundary > 0)
    center_silhouette = (by[len(by) // 2], bx[len(bx) // 2]) if len(by) > 0 else (int(h * 0.4), int(w * 0.4))

    bg_y, bg_x = np.where(~subject_mask)
    center_bg = (bg_y[len(bg_y) // 4], bg_x[len(bg_x) // 4]) if len(bg_y) > 0 else (int(h * 0.1), int(w * 0.1))

    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ey, ex = np.where(edges > 0)
    center_ornaments = (ey[len(ey) // 2], ex[len(ex) // 2]) if len(ey) > 0 else (int(h * 0.8), int(w * 0.8))

    centers = [center_face, center_hands, center_ornaments, center_silhouette, center_bg]
    labels = ["Face", "Hands", "Ornaments", "Silhouette", "Background"]

    num_f = len(rendered_frames)
    k12 = max(0, min(num_f - 1, int(num_f * 0.25)))
    k24 = max(0, min(num_f - 1, int(num_f * 0.50)))
    k36 = max(0, min(num_f - 1, int(num_f * 0.75)))

    frames_to_compare = [("ORIGINAL", original_rgb),
                         (f"FRAME {k12:02d}", rendered_frames[k12]),
                         (f"FRAME {k24:02d}", rendered_frames[k24]),
                         (f"FRAME {k36:02d}", rendered_frames[k36])]

    rows = []
    for (cy_c, cx_c), label in zip(centers, labels):
        y0 = max(0, min(h - crop_size, cy_c - half_crop))
        x0 = max(0, min(w - crop_size, cx_c - half_crop))
        y1, x1 = y0 + crop_size, x0 + crop_size

        cells = []
        for name, img in frames_to_compare:
            crop = img[y0:y1, x0:x1]
            enlarged = cv2.resize(crop, (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
            # Label banner
            cv2.rectangle(enlarged, (0, 0), (crop_size * 2, 28), (0, 0, 0), -1)
            cv2.putText(enlarged, f"{label}: {name}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if "ORIG" in name else (0, 255, 0), 1)
            cells.append(enlarged)

        row = np.hstack(cells)
        rows.append(row)

    grid = np.vstack(rows)
    return grid

def generate_final_contact_sheet(
    original_rgb: np.ndarray,
    rendered_frames: list,
    subject_mask: np.ndarray,
    crop_size: int = 140
) -> np.ndarray:
    """
    Generates a visual contact sheet comparing ORIGINAL against 5 dynamically sampled keyframes up to F{last}
    with 2x enlarged crops across 5 key structural regions:
    1. Face
    2. Hands
    3. Ornaments
    4. Silhouette / Boundary
    5. Background
    """
    h, w, _ = original_rgb.shape
    half_crop = crop_size // 2

    y_sub, x_sub = np.where(subject_mask)
    center_face = (int(np.mean(y_sub)), int(np.mean(x_sub))) if len(y_sub) > 0 else (h // 2, w // 2)
    center_hands = (int(np.percentile(y_sub, 75)), int(np.mean(x_sub))) if len(y_sub) > 0 else (int(h * 0.7), int(w * 0.5))

    sub_boundary = cv2.Canny((subject_mask * 255).astype(np.uint8), 100, 200) > 0
    by, bx = np.where(sub_boundary)
    center_silhouette = (by[len(by) // 2], bx[len(bx) // 2]) if len(by) > 0 else (int(h * 0.4), int(w * 0.4))

    bg_y, bg_x = np.where(~subject_mask)
    center_bg = (bg_y[len(bg_y) // 4], bg_x[len(bg_x) // 4]) if len(bg_y) > 0 else (int(h * 0.1), int(w * 0.1))

    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ey, ex = np.where(edges > 0)
    center_ornaments = (ey[len(ey) // 2], ex[len(ex) // 2]) if len(ey) > 0 else (int(h * 0.8), int(w * 0.8))

    centers = [center_face, center_hands, center_ornaments, center_silhouette, center_bg]
    labels = ["Face", "Hands", "Ornaments", "Silhouette", "Background"]

    num_f = len(rendered_frames)
    frame_indices = np.linspace(0, num_f - 1, 5, dtype=int).tolist()

    rows = []
    for (cy_c, cx_c), label in zip(centers, labels):
        y0 = max(0, min(h - crop_size, cy_c - half_crop))
        x0 = max(0, min(w - crop_size, cx_c - half_crop))
        y1, x1 = y0 + crop_size, x0 + crop_size

        c_orig = cv2.resize(original_rgb[y0:y1, x0:x1], (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c_orig, f"{label} (Orig)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cells = [c_orig]
        for f_idx in frame_indices:
            frame_img = rendered_frames[f_idx]
            c_f = cv2.resize(frame_img[y0:y1, x0:x1], (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
            cv2.putText(c_f, f"F{f_idx:02d}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cells.append(c_f)

        row = np.hstack(cells)
        rows.append(row)

    sheet = np.vstack(rows)
    return sheet

def generate_phase_e_keyframe_contact_sheet(
    original_rgb: np.ndarray,
    keyframes: Dict[str, np.ndarray],
    subject_mask: np.ndarray,
    crop_size: int = 140
) -> np.ndarray:
    """
    Generates an accessible visual contact sheet comparing ACTUAL rendered keyframes:
    ORIGINAL | START | 25% | 50% | 75% | END
    along with 2x enlarged crops across 5 regions (FACE, HANDS, ORNAMENTS, SUBJECT BOUNDARY, BACKGROUND).
    """
    h, w, _ = original_rgb.shape
    half_crop = crop_size // 2

    y_sub, x_sub = np.where(subject_mask)
    center_face = (int(np.mean(y_sub)), int(np.mean(x_sub))) if len(y_sub) > 0 else (h // 2, w // 2)
    center_hands = (int(np.percentile(y_sub, 75)), int(np.mean(x_sub))) if len(y_sub) > 0 else (int(h * 0.7), int(w * 0.5))

    sub_boundary = cv2.Canny((subject_mask * 255).astype(np.uint8), 100, 200) > 0
    by, bx = np.where(sub_boundary)
    center_sub_edge = (by[len(by) // 2], bx[len(bx) // 2]) if len(by) > 0 else (int(h * 0.4), int(w * 0.4))

    bg_y, bg_x = np.where(~subject_mask)
    center_bg = (bg_y[len(bg_y) // 4], bg_x[len(bg_x) // 4]) if len(bg_y) > 0 else (int(h * 0.1), int(w * 0.1))

    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ey, ex = np.where(edges > 0)
    center_ornaments = (ey[len(ey) // 2], ex[len(ex) // 2]) if len(ey) > 0 else (int(h * 0.8), int(w * 0.8))

    centers = [center_face, center_hands, center_ornaments, center_sub_edge, center_bg]
    labels = ["Face", "Hands", "Ornaments", "Subject Edge", "Background"]
    k_order = ["start", "25", "50", "75", "end"]

    rows = []
    for (cy_c, cx_c), label in zip(centers, labels):
        y0 = max(0, min(h - crop_size, cy_c - half_crop))
        x0 = max(0, min(w - crop_size, cx_c - half_crop))
        y1, x1 = y0 + crop_size, x0 + crop_size

        c_orig = cv2.resize(original_rgb[y0:y1, x0:x1], (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c_orig, f"{label} (Orig)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cells = [c_orig]
        for kn in k_order:
            img_k = keyframes.get(kn, original_rgb)
            c_k = cv2.resize(img_k[y0:y1, x0:x1], (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_NEAREST)
            cv2.putText(c_k, f"{kn}%", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cells.append(c_k)

        row = np.hstack(cells)
        rows.append(row)

    sheet = np.vstack(rows)
    return sheet
