import cv2
import numpy as np
from typing import Optional, Tuple, Any
from geometry.projection import back_project_points, project_3d_points
from geometry.transforms import transform_3d_points

def construct_layer_motion_map(
    shape: Tuple[int, int],
    subject_mask: np.ndarray,
    spatial_diagnostics: Optional[Any] = None,
    motion_amplitude: str = "MEDIUM"
) -> np.ndarray:
    """
    Constructs a 2D float32 layer motion multiplier map m(u, v).
    Uses layer-differentiated motion multipliers based on motion_amplitude ("LOW", "MEDIUM", "HIGH").
    """
    from spatial_intelligence.camera_model import compute_layer_motion_multiplier

    h, w = shape
    default_bg_mult = compute_layer_motion_multiplier("BACKGROUND", motion_amplitude)
    motion_map = np.full((h, w), fill_value=default_bg_mult, dtype=np.float32)

    sub_mult = compute_layer_motion_multiplier("PRIMARY_SUBJECT", motion_amplitude)

    if spatial_diagnostics is not None and hasattr(spatial_diagnostics, "scene_graph"):
        for ent in spatial_diagnostics.scene_graph.entities.values():
            role_str = str(ent.layer_role.value if hasattr(ent.layer_role, "value") else ent.layer_role).upper()
            if role_str == "ANALYSIS_ONLY":
                continue
            mult = compute_layer_motion_multiplier(role_str, motion_amplitude)
            motion_map[ent.mask] = mult

    # Guarantee primary subject receives primary subject multiplier
    motion_map[subject_mask] = sub_mult

    return motion_map

def render_single_frame_forward_splatting(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    bg_plate: np.ndarray,
    bg_depth: np.ndarray,
    provenance_map: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_discontinuity_threshold: float = 0.5,
    layer_motion_map: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Renders a synthesized view using forward subpixel splatting and deterministic Z-buffering.
    Features:
    1. Independent forward splatting for background plate and foreground surfaces.
    2. Back-projects foreground and background surfaces to 3D.
    3. Applies camera transformation P' = R @ P + t.
    4. Subpixel splatting with bilinear distribution to 2x2 target pixel neighborhood.
    5. Depth discontinuity protection: suppresses splatting across large depth jumps to avoid rubber-sheet stretching.
    6. Deterministic Z-buffer compositing: foreground layer strictly overwrites background layer where valid foreground splats exist.

    Returns: (synthesized_rgb, rendered_depth_buffer, output_provenance)
    """
    height, width, _ = rgb_array.shape

    # 1. Prepare source grids
    u_grid, v_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()

    # Compute 2D depth gradients for discontinuity detection
    d_grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    d_grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    d_grad_mag = np.sqrt(d_grad_x**2 + d_grad_y**2).ravel()
    is_discontinuity = d_grad_mag > depth_discontinuity_threshold

    def splat_layer(color_src: np.ndarray, depth_src: np.ndarray, prov_src: np.ndarray, mult_src: Optional[np.ndarray], is_fg: bool = False):
        z_buf = np.full((height, width), fill_value=1e9, dtype=np.float32)
        accum_col = np.zeros((height, width, 3), dtype=np.float32)
        accum_w = np.zeros((height, width), dtype=np.float32)
        out_prov = np.zeros((height, width), dtype=np.float32)

        colors_flat = color_src.reshape(-1, 3).astype(np.float32)
        depths_flat = depth_src.ravel()
        prov_flat = prov_src.ravel()

        pts_3d = back_project_points(u_flat, v_flat, depths_flat, fx, fy, cx, cy)
        if mult_src is not None:
            t_pixel = t[None, :] * mult_src.ravel()[:, None]
            pts_trans = (pts_3d @ R.T) + t_pixel
        else:
            pts_trans = transform_3d_points(pts_3d, R, t)

        proj_u, proj_v, proj_z = project_3d_points(pts_trans, fx, fy, cx, cy)
        valid_mask = (proj_z > 0.05) & (proj_u >= 0.0) & (proj_u < width - 1) & (proj_v >= 0.0) & (proj_v < height - 1)
        if is_fg:
            valid_mask = valid_mask & (~is_discontinuity)

        valid_indices = np.where(valid_mask)[0]
        sort_order = np.argsort(-proj_z[valid_indices])
        sorted_indices = valid_indices[sort_order]

        pu = proj_u[sorted_indices]
        pv = proj_v[sorted_indices]
        pz = proj_z[sorted_indices]
        pcol = colors_flat[sorted_indices]
        pprov = prov_flat[sorted_indices]

        u0 = np.floor(pu).astype(int)
        v0 = np.floor(pv).astype(int)
        u1 = u0 + 1
        v1 = v0 + 1

        du = (pu - u0).astype(np.float32)
        dv = (pv - v0).astype(np.float32)

        subpixel_offsets = [
            ((1.0 - du) * (1.0 - dv), u0, v0),
            (du * (1.0 - dv), u1, v0),
            ((1.0 - du) * dv, u0, v1),
            (du * dv, u1, v1)
        ]

        for w_arr, u_arr, v_arr in subpixel_offsets:
            valid_sub = (w_arr > 1e-4) & (u_arr >= 0) & (u_arr < width) & (v_arr >= 0) & (v_arr < height)
            if not np.any(valid_sub):
                continue

            u_sub = u_arr[valid_sub]
            v_sub = v_arr[valid_sub]
            w_sub = w_arr[valid_sub]
            z_sub = pz[valid_sub]
            col_sub = pcol[valid_sub]
            prov_sub = pprov[valid_sub]

            curr_z_vals = z_buf[v_sub, u_sub]
            closer_mask = z_sub < (curr_z_vals - 0.001)
            if np.any(closer_mask):
                u_c, v_c = u_sub[closer_mask], v_sub[closer_mask]
                z_buf[v_c, u_c] = z_sub[closer_mask]
                accum_col[v_c, u_c] = 0.0
                accum_w[v_c, u_c] = 0.0

            curr_z_updated = z_buf[v_sub, u_sub]
            visible_mask = z_sub <= (curr_z_updated + 0.001)
            if not np.any(visible_mask):
                continue

            u_vis = u_sub[visible_mask]
            v_vis = v_sub[visible_mask]
            w_vis = w_sub[visible_mask]
            z_vis = z_sub[visible_mask]
            col_vis = col_sub[visible_mask]
            prov_vis = prov_sub[visible_mask]

            np.minimum.at(z_buf, (v_vis, u_vis), z_vis)
            np.add.at(accum_col, (v_vis, u_vis), col_vis * w_vis[:, None])
            np.add.at(accum_w, (v_vis, u_vis), w_vis)
            out_prov[v_vis, u_vis] = prov_vis

        return z_buf, accum_col, accum_w, out_prov

    # Render Background Layer
    bg_z, bg_col, bg_w, bg_p = splat_layer(bg_plate, bg_depth, provenance_map, layer_motion_map, is_fg=False)

    # Render Foreground Layer
    fg_z, fg_col, fg_w, fg_p = splat_layer(rgb_array, depth_map, provenance_map, layer_motion_map, is_fg=True)

    # Composite layers: Where foreground splats exist (fg_w > 0), foreground wins
    fg_mask = fg_w > 0.05
    syn_rgb = np.zeros((height, width, 3), dtype=np.float32)
    rendered_z = bg_z.copy()
    output_prov = bg_p.copy()

    # Background layer synthesis
    bg_valid = bg_w > 0
    syn_rgb[bg_valid] = bg_col[bg_valid] / bg_w[bg_valid][..., None]

    # Inpaint or fill background disocclusion holes
    if not np.all(bg_valid):
        hole_mask = (~bg_valid).astype(np.uint8) * 255
        syn_uint8 = np.clip(syn_rgb, 0, 255).astype(np.uint8)
        # Check if zero camera translation (identity render)
        if np.max(np.abs(t)) < 1e-4 and abs(R[0, 0] - 1.0) < 1e-4:
            syn_rgb[~bg_valid] = bg_plate[~bg_valid].astype(np.float32)
        else:
            inpainted = cv2.inpaint(syn_uint8, hole_mask, 3, cv2.INPAINT_TELEA)
            syn_rgb[~bg_valid] = inpainted[~bg_valid].astype(np.float32)

    # Foreground layer overlay
    syn_rgb[fg_mask] = fg_col[fg_mask] / fg_w[fg_mask][..., None]
    rendered_z[fg_mask] = fg_z[fg_mask]
    output_prov[fg_mask] = fg_p[fg_mask]

    syn_rgb = np.clip(syn_rgb, 0.0, 255.0).astype(np.uint8)
    return syn_rgb, rendered_z, output_prov
