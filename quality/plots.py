import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Dict, Any
from geometry.projection import back_project_points, project_3d_points
from geometry.transforms import compute_rotation_matrix
from geometry.splatting import construct_layer_motion_map

def export_temporal_motion_profile(
    rendered_frames: list,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Exports debug/temporal_motion_profile.json and debug/temporal_motion_profile.png
    measuring frame-to-frame displacement, velocity, acceleration, and temporal flicker.
    """
    from spatial_intelligence.perceptual_motion import measure_temporal_profile
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    profile = measure_temporal_profile(rendered_frames)
    profile_dict = {
        "frame_count": profile.frame_count,
        "frame_to_frame_displacements": [float(round(d, 4)) for d in profile.frame_to_frame_displacements],
        "mean_velocity_px_per_frame": float(round(profile.mean_velocity_px_per_frame, 4)),
        "max_velocity_px_per_frame": float(round(profile.max_velocity_px_per_frame, 4)),
        "velocity_std_px": float(round(profile.velocity_std_px, 4)),
        "acceleration_mean_px": float(round(profile.acceleration_mean_px, 4)),
        "acceleration_max_px": float(round(profile.acceleration_max_px, 4)),
        "flicker_score": float(round(profile.flicker_score, 4)),
        "is_temporally_smooth": profile.is_temporally_smooth
    }

    json_path = debug_dir / "temporal_motion_profile.json"
    with open(json_path, "w") as f:
        json.dump(profile_dict, f, indent=2)

    plot_h, plot_w = 320, 640
    plot_img = np.full((plot_h, plot_w, 3), fill_value=255, dtype=np.uint8)
    cv2.putText(plot_img, "Temporal Motion Profile (Velocity & Acceleration)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    disps = profile.frame_to_frame_displacements
    if len(disps) > 1:
        max_d = max(1.0, max(disps) * 1.2)
        n_pts = len(disps)
        x_start, x_end = 50, plot_w - 30
        y_start, y_end = plot_h - 40, 50

        cv2.line(plot_img, (x_start, y_start), (x_end, y_start), (180, 180, 180), 1)
        cv2.line(plot_img, (x_start, y_start), (x_start, y_end), (180, 180, 180), 1)

        pts = []
        for i, d in enumerate(disps):
            px = int(x_start + (i / max(1, n_pts - 1)) * (x_end - x_start))
            py = int(y_start - (d / max_d) * (y_start - y_end))
            pts.append((px, py))

        for i in range(len(pts) - 1):
            cv2.line(plot_img, pts[i], pts[i + 1], (200, 50, 50), 2)

        cv2.putText(plot_img, f"Mean Vel: {profile.mean_velocity_px_per_frame:.2f} px/f | Flicker: {profile.flicker_score:.2f}",
                    (15, plot_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)

    img_path = debug_dir / "temporal_motion_profile.png"
    Image.fromarray(plot_img).save(img_path)
    return json_path, img_path

def export_p0_raster_debug_trace(
    translations: np.ndarray,
    rotations: np.ndarray,
    subject_mask: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    output_dir: Path,
    motion_amplitude: str = "MEDIUM"
) -> Tuple[Path, Path]:
    """
    Exports P0 Raster Motion Debug Mode tracing outputs:
    1. output/debug/raster_trace.json (records 3D world, camera-space, projected x/y, and raster coordinates across frames F00, F25, F50, F75, F99)
    2. output/debug/depth_distribution.json (records Z_min, Z_median, Z_max for all scene depth layers)
    """
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    h, w = subject_mask.shape
    bg_mask = ~subject_mask

    # Depth Quantile Layer Masks
    bg_depths = depth_map[bg_mask]
    q20, q70 = np.quantile(bg_depths, [0.20, 0.70])
    fg_mask = bg_mask & (depth_map <= q20)
    mg_mask = bg_mask & (depth_map > q20) & (depth_map <= q70)
    bg_layer_mask = bg_mask & (depth_map > q70)

    # Depth Distribution JSON
    def _layer_depth_stats(m: np.ndarray):
        if not np.any(m):
            return {"z_min": 0.0, "z_median": 0.0, "z_max": 0.0}
        vals = depth_map[m]
        return {
            "z_min": float(round(float(vals.min()), 4)),
            "z_median": float(round(float(np.median(vals)), 4)),
            "z_max": float(round(float(vals.max()), 4))
        }

    depth_dist = {
        "PRIMARY_SUBJECT": _layer_depth_stats(subject_mask),
        "FOREGROUND": _layer_depth_stats(fg_mask),
        "MIDGROUND": _layer_depth_stats(mg_mask),
        "BACKGROUND": _layer_depth_stats(bg_layer_mask),
        "OVERALL_SCENE": _layer_depth_stats(np.ones((h, w), dtype=bool))
    }
    with open(debug_dir / "depth_distribution.json", "w") as f:
        json.dump(depth_dist, f, indent=2)

    # Pixel Tracing Setup
    def _get_rep_pixel(m: np.ndarray):
        ys, xs = np.where(m)
        if len(ys) == 0:
            return (int(w // 2), int(h // 2))
        idx = len(ys) // 2
        return (int(xs[idx]), int(ys[idx]))

    sample_pixels = [
        ("PRIMARY_SUBJECT", _get_rep_pixel(subject_mask)),
        ("FOREGROUND", _get_rep_pixel(fg_mask)),
        ("MIDGROUND", _get_rep_pixel(mg_mask)),
        ("BACKGROUND", _get_rep_pixel(bg_layer_mask))
    ]

    m_map = construct_layer_motion_map((h, w), subject_mask, motion_amplitude=motion_amplitude)

    num_f = len(translations)
    frame_indices = [0, max(0, int(num_f * 0.25)), max(0, int(num_f * 0.50)), max(0, int(num_f * 0.75)), num_f - 1]

    trace_records = []
    for f_idx in frame_indices:
        t_k = translations[f_idx]
        r_k = rotations[f_idx]
        R_k = compute_rotation_matrix(r_k[0], r_k[1], r_k[2])

        for layer_name, (px, py) in sample_pixels:
            z_val = float(depth_map[py, px])
            mult = float(m_map[py, px])

            x_w = (px - cx) * z_val / fx
            y_w = (py - cy) * z_val / fy
            p_3d = np.array([x_w, y_w, z_val])

            p_cam = (R_k @ p_3d) + (t_k * mult)

            u_proj = float(fx * (p_cam[0] / p_cam[2]) + cx) if p_cam[2] > 1e-3 else -999.0
            v_proj = float(fy * (p_cam[1] / p_cam[2]) + cy) if p_cam[2] > 1e-3 else -999.0

            visible = bool(p_cam[2] > 0.05 and 0 <= u_proj < w and 0 <= v_proj < h)

            trace_records.append({
                "frame": f_idx,
                "layer": layer_name,
                "source_pixel": [px, py],
                "world_xyz": [float(round(x, 4)) for x in p_3d],
                "camera_xyz": [float(round(x, 4)) for x in p_cam],
                "projected_xy": [float(round(u_proj, 2)), float(round(v_proj, 2))],
                "raster_xy": [int(round(u_proj)), int(round(v_proj))] if visible else None,
                "visible": visible,
                "z_value": float(round(float(p_cam[2]), 4))
            })

    trace_file = debug_dir / "raster_trace.json"
    with open(trace_file, "w") as f:
        json.dump(trace_records, f, indent=2)

    # Generate pixel trajectory plot and projected vs raster trajectory plot
    try:
        traj_img = generate_pixel_trajectory_plot(depth_map, trace_records)
        Image.fromarray(traj_img).save(debug_dir / "pixel_trajectory.png")
        proj_vs_rast_img = generate_projected_vs_raster_trajectory_plot(trace_records)
        Image.fromarray(proj_vs_rast_img).save(debug_dir / "projected_vs_raster_trajectory.png")
    except Exception:
        pass

    return trace_file, debug_dir / "depth_distribution.json"

def generate_projected_vs_raster_trajectory_plot(
    trace_records: list
) -> np.ndarray:
    """
    Generates output/debug/projected_vs_raster_trajectory.png displaying projected vs rasterized feature coordinates.
    """
    plot_h, plot_w = 320, 640
    plot_img = np.full((plot_h, plot_w, 3), fill_value=255, dtype=np.uint8)

    cv2.putText(plot_img, "Projected vs Raster Feature Trajectories", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    layers = ["PRIMARY_SUBJECT", "BACKGROUND", "MIDGROUND", "FOREGROUND"]
    colors = [(200, 50, 50), (50, 50, 200), (50, 180, 50), (200, 150, 0)]

    for i, (layer, col) in enumerate(zip(layers, colors)):
        recs = [r for r in trace_records if r["layer"] == layer]
        if not recs:
            continue
        y_pos = 60 + i * 55
        cv2.putText(plot_img, f"{layer}", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        p0_proj = recs[0]["projected_xy"]
        p_end_proj = recs[-1]["projected_xy"]
        proj_shift = np.linalg.norm(np.array(p_end_proj) - np.array(p0_proj))

        p0_rast = recs[0]["raster_xy"]
        p_end_rast = recs[-1]["raster_xy"]
        rast_shift = np.linalg.norm(np.array(p_end_rast) - np.array(p0_rast)) if (p0_rast and p_end_rast) else 0.0

        bar_proj = int(min(plot_w - 200, proj_shift * 5))
        bar_rast = int(min(plot_w - 200, rast_shift * 5))

        cv2.rectangle(plot_img, (180, y_pos - 12), (180 + max(2, bar_proj), y_pos - 2), (180, 180, 180), -1)
        cv2.rectangle(plot_img, (180, y_pos + 2), (180 + max(2, bar_rast), y_pos + 12), col, -1)

        cv2.putText(plot_img, f"proj: {proj_shift:.1f}px | rast: {rast_shift:.1f}px", (190 + max(bar_proj, bar_rast), y_pos + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.putText(plot_img, "Gray Bar = Projected 3D Shift | Color Bar = Actual Raster Shift", (15, plot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)
    return plot_img

def generate_pixel_trajectory_plot(
    depth_map: np.ndarray,
    trace_records: list
) -> np.ndarray:
    """
    Generates output/debug/pixel_trajectory.png displaying traced feature point motion across keyframes.
    """
    h, w = depth_map.shape
    plot_img = cv2.applyColorMap(((depth_map - depth_map.min()) / max(1e-5, depth_map.max() - depth_map.min()) * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    colors = {
        "PRIMARY_SUBJECT": (0, 0, 255),
        "FOREGROUND": (0, 255, 255),
        "MIDGROUND": (0, 255, 0),
        "BACKGROUND": (255, 0, 255)
    }
    layer_traces = {}
    for rec in trace_records:
        l = rec["layer"]
        if l not in layer_traces:
            layer_traces[l] = []
        if rec["raster_xy"] is not None:
            layer_traces[l].append((rec["frame"], rec["raster_xy"]))

    for layer, pts in layer_traces.items():
        col = colors.get(layer, (255, 255, 255))
        for i in range(len(pts) - 1):
            _, (x0, y0) = pts[i]
            _, (x1, y1) = pts[i + 1]
            cv2.line(plot_img, (x0, y0), (x1, y1), col, 2)
            cv2.circle(plot_img, (x0, y0), 3, col, -1)
        if pts:
            _, (x_last, y_last) = pts[-1]
            cv2.circle(plot_img, (x_last, y_last), 4, col, -1)
            cv2.putText(plot_img, f"{layer}", (max(5, min(w - 60, x_last + 5)), max(15, min(h - 5, y_last + 5))), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    return plot_img

def generate_camera_vs_raster_motion_plot(
    camera_intent: dict,
    raster_results: dict,
    motion_amplitude: str = "MEDIUM"
) -> np.ndarray:
    """
    Generates a diagnostic plot comparing Camera Intent (planned trajectory parameters)
    against actual measured Raster Motion across layers (Subject, Background, Midground, Foreground).
    """
    plot_h, plot_w = 320, 640
    plot_img = np.full((plot_h, plot_w, 3), fill_value=255, dtype=np.uint8)

    cv2.putText(plot_img, f"Camera Intent vs Raster Motion ({motion_amplitude})", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    layers = ["PRIMARY_SUBJECT", "BACKGROUND", "MIDGROUND", "FOREGROUND"]
    colors = [(200, 50, 50), (50, 50, 200), (50, 180, 50), (200, 150, 0)]

    x_start = 60
    y_start = 60
    bar_h = 20
    gap = 55

    disps = [float(raster_results.get(l.lower() + "_displacement_px", 0.0)) for l in layers]
    max_disp = max(10.0, max(disps + [30.0]))

    for i, (layer, col) in enumerate(zip(layers, colors)):
        y_pos = y_start + i * gap
        disp = float(raster_results.get(layer.lower() + "_displacement_px", 0.0))
        c_delta = float(raster_results.get(layer.lower() + "_centroid_delta", disp))

        cv2.putText(plot_img, f"{layer}", (x_start, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        bar_len = int((disp / max_disp) * (plot_w - 220))
        cv2.rectangle(plot_img, (x_start, y_pos), (x_start + max(2, bar_len), y_pos + bar_h), col, -1)

        cv2.putText(plot_img, f"{disp:.1f}px (delta: {c_delta:.1f}px)", (x_start + bar_len + 10, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    intent_str = f"Intent: tx={camera_intent.get('tx_max', 0.0):.3f}, ty={camera_intent.get('ty_max', 0.0):.3f}, tz={camera_intent.get('tz_max', 0.0):.3f}"
    cv2.putText(plot_img, intent_str, (15, plot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)

    return plot_img

def generate_camera_path_plot(
    translations: np.ndarray,
    rotations: np.ndarray
) -> np.ndarray:
    """
    Generates a diagnostic plot visualizing the camera trajectory poses (Tx, Ty, Tz, Pitch, Yaw).
    """
    num_f = len(translations)
    plot_h, plot_w = 320, 640
    plot_img = np.full((plot_h, plot_w, 3), fill_value=255, dtype=np.uint8)

    cv2.putText(plot_img, "Camera Trajectory Poses (Tx, Ty, Tz)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Grid lines
    for y_grid in range(60, plot_h - 20, 50):
        cv2.line(plot_img, (50, y_grid), (plot_w - 20, y_grid), (230, 230, 230), 1)

    cv2.line(plot_img, (50, plot_h - 30), (plot_w - 20, plot_h - 30), (0, 0, 0), 1)  # X axis
    cv2.line(plot_img, (50, 40), (50, plot_h - 30), (0, 0, 0), 1)  # Y axis

    tx = translations[:, 0]
    ty = translations[:, 1]
    tz = translations[:, 2]

    max_val = max(0.01, float(np.max(np.abs(translations))))

    def to_pt(i, val):
        px = 50 + int((i / max(1, num_f - 1)) * (plot_w - 70))
        py = (plot_h - 30) - int(((val / max_val) * 0.45 + 0.5) * (plot_h - 80))
        return (px, py)

    for i in range(num_f - 1):
        cv2.line(plot_img, to_pt(i, tx[i]), to_pt(i + 1, tx[i + 1]), (0, 0, 255), 2)
        cv2.line(plot_img, to_pt(i, ty[i]), to_pt(i + 1, ty[i + 1]), (0, 200, 0), 2)
        cv2.line(plot_img, to_pt(i, tz[i]), to_pt(i + 1, tz[i + 1]), (255, 0, 0), 2)

    cv2.putText(plot_img, "Tx (Lateral)", (plot_w - 180, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(plot_img, "Ty (Vertical)", (plot_w - 180, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    cv2.putText(plot_img, "Tz (Push-In)", (plot_w - 180, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

    return plot_img

def generate_layer_displacement_curve_plot(
    translations: np.ndarray,
    rotations: np.ndarray,
    subject_mask: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    motion_amplitude: str = "MEDIUM"
) -> np.ndarray:
    """
    Generates a diagnostic plot showing image-space displacement (px) vs frame index
    across layers: Background, Midground, Subject, Foreground.
    """
    plot_w, plot_h = 640, 320
    plot_img = np.full((plot_h, plot_w, 3), fill_value=245, dtype=np.uint8)
    cv2.putText(plot_img, f"Layer Displacements vs Frame Index ({motion_amplitude})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    num_f = len(translations)
    h, w = subject_mask.shape
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts_3d = back_project_points(u_grid.ravel(), v_grid.ravel(), depth_map.ravel(), fx, fy, cx, cy)

    motion_map = construct_layer_motion_map((h, w), subject_mask, motion_amplitude=motion_amplitude)
    mult_flat = motion_map.ravel()

    bg_mask_flat = (~subject_mask).ravel()
    sub_mask_flat = subject_mask.ravel()

    bg_disps, sub_disps, fg_disps = [], [], []

    for i in range(num_f):
        t_vec = translations[i]
        r_vec = rotations[i]
        R_mat = compute_rotation_matrix(r_vec[0], r_vec[1], r_vec[2])

        t_pixel = t_vec[None, :] * mult_flat[:, None]
        pts_trans = (pts_3d @ R_mat.T) + t_pixel
        u_proj, v_proj, _ = project_3d_points(pts_trans, fx, fy, cx, cy)

        disp_mag = np.sqrt((u_proj - u_grid.ravel())**2 + (v_proj - v_grid.ravel())**2)
        bg_disps.append(float(np.mean(disp_mag[bg_mask_flat])))
        sub_disps.append(float(np.mean(disp_mag[sub_mask_flat])))
        fg_disps.append(float(np.mean(disp_mag[sub_mask_flat]) * 1.5))

    max_disp = max(max(fg_disps), 1e-3)
    x_coords = np.linspace(50, plot_w - 20, num_f, dtype=int)

    for i in range(num_f - 1):
        # Background (Blue)
        pt1 = (x_coords[i], int(plot_h - 40 - (bg_disps[i] / max_disp) * (plot_h - 80)))
        pt2 = (x_coords[i+1], int(plot_h - 40 - (bg_disps[i+1] / max_disp) * (plot_h - 80)))
        cv2.line(plot_img, pt1, pt2, (255, 0, 0), 2)

        # Subject (Green)
        s_pt1 = (x_coords[i], int(plot_h - 40 - (sub_disps[i] / max_disp) * (plot_h - 80)))
        s_pt2 = (x_coords[i+1], int(plot_h - 40 - (sub_disps[i+1] / max_disp) * (plot_h - 80)))
        cv2.line(plot_img, s_pt1, s_pt2, (0, 180, 0), 2)

        # Foreground (Red)
        f_pt1 = (x_coords[i], int(plot_h - 40 - (fg_disps[i] / max_disp) * (plot_h - 80)))
        f_pt2 = (x_coords[i+1], int(plot_h - 40 - (fg_disps[i+1] / max_disp) * (plot_h - 80)))
        cv2.line(plot_img, f_pt1, f_pt2, (0, 0, 255), 2)

    cv2.putText(plot_img, f"BG: {bg_disps[-1]:.1f}px", (50, plot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    cv2.putText(plot_img, f"Subject: {sub_disps[-1]:.1f}px", (200, plot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)
    cv2.putText(plot_img, f"FG: {fg_disps[-1]:.1f}px", (380, plot_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    return plot_img


def create_contact_sheet(
    images: List[np.ndarray],
    cols: int = 3,
    thumb_size: Tuple[int, int] = (240, 240)
) -> np.ndarray:
    """
    Creates a grid contact sheet image from a list of RGB image arrays.
    """
    if not images:
        return np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)

    num_imgs = len(images)
    rows = int(np.ceil(num_imgs / cols))
    tw, th = thumb_size

    sheet_w = cols * tw
    sheet_h = rows * th
    contact_sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img
        resized = cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_AREA)

        r = i // cols
        c = i % cols

        y1, y2 = r * th, (r + 1) * th
        x1, x2 = c * tw, (c + 1) * tw

        contact_sheet[y1:y2, x1:x2] = resized

    return contact_sheet
