# ROOT CAUSE AUDIT REPORT — PERCEPTUAL CAMERA MOTION ATTENUATION

## Executive Summary
This audit traces the complete motion execution chain in `v0_pipeline.py`, `spatial_intelligence/`, and `subject_selection/` to identify why mathematical camera trajectories failed to produce visually obvious cinematic camera movement in rendered MP4 outputs.

---

## 1. End-to-End Motion Chain Trace

```
INPUT IMAGE (W x H)
    ↓
DEPTH INFERENCE (Depth Anything V2) → continuous normalized depth map Z in [z_min, z_max]
    ↓
SUBJECT SELECTION (SAM 2) → primary_subject_mask
    ↓
SPATIAL INTELLIGENCE → depth field, layer masks (PRIMARY_SUBJECT, FOREGROUND, MIDGROUND, BACKGROUND)
    ↓
CAMERA TRAJECTORY (generate_c1_smooth_trajectory) → translations t = [tx, ty, tz], rotations R = [pitch, yaw, roll]
    ↓
LAYER MULTIPLIER MAP (construct_layer_motion_map) → m(u, v) per pixel
    ↓
3D BACK-PROJECTION (back_project_points) → P = [(u-cx)*Z/fx, (v-cy)*Z/fy, Z]
    ↓
3D TRANSFORMATION → P' = R @ P + (t * m(u, v))
    ↓
3D PROJECTION (project_3d_points) → u' = fx * X'/Z' + cx, v' = fy * Y'/Z' + cy
    ↓
FORWARD SUBPIXEL SPLATTING & Z-BUFFERING → rendered_frames
    ↓
RASTER MOTION MEASUREMENT (compute_perceptual_motion_score) → Farneback optical flow & pixel diff
    ↓
QUALITY GATES & MP4 ENCODING
```

---

## 2. Mathematical Root Cause of 0.37 px Displacement

### The DIBR Perspective Scale Equation
Given a 3D point $P = [X, Y, Z]^T$ back-projected from image pixel $(u, v)$ with camera principal point $(c_x, c_y)$ and focal length $f_x$:
$$X = \frac{(u - c_x) Z}{f_x}, \quad Y = \frac{(v - c_y) Z}{f_y}$$

Under a pure forward push-in camera translation $t_z < 0$, the transformed Z coordinate becomes $Z' = Z + t_z$.
The reprojected 2D pixel coordinate $u'$ is:
$$u' = f_x \frac{X}{Z'} + c_x = f_x \frac{(u - c_x) Z / f_x}{Z + t_z} + c_x = c_x + (u - c_x) \frac{Z}{Z + t_z}$$

The resulting 2D raster displacement $\Delta u = u' - u$ is:
$$\Delta u = (u - c_x) \left( \frac{Z}{Z + t_z} - 1 \right) = (u - c_x) \frac{-t_z}{Z + t_z}$$

### Why 0.37 px Occurred
1. **Low Image Resolution & Centered Points**: On a $320 \times 320$ image ($c_x = 160, f_x = 320$), a background feature near the subject center at $u = 180$ ($u - c_x = 20\text{px}$) with depth $Z = 5.0$ and raw $t_z = -0.08$ produces:
$$\Delta u = 20 \cdot \frac{0.08}{5.0 - 0.08} = 20 \cdot \frac{0.08}{4.92} = 0.325\text{px}$$
2. **Mean Signed Optical Flow Cancellation**: Averaging signed flow vectors $(u, v)$ across symmetric layer regions caused positive (+0.325px) and negative (-0.325px) motion vectors on opposing sides of the optical axis to cancel out to near-zero ($0.37\text{px}$).

---

## 3. Camera Conventions Audit

| Parameter | Convention / Formula | Audit Status |
| :--- | :--- | :--- |
| **Coordinate System** | Right-handed pinhole: $+X$ right, $+Y$ down, $+Z$ into scene | Verified |
| **Push-In Direction** | Negative $t_z$ decreases camera-to-subject distance | Verified |
| **Focal Length** | $f_x = f_y = \max(W, H)$ (~53° FOV) | Verified |
| **Principal Point** | $c_x = W / 2, c_y = H / 2$ | Verified |
| **Pixel Centers** | Pixel grid integer bounds $0 \le u < W, 0 \le v < H$ | Verified |
| **Depth Scale** | Normalized continuous depth $Z \in [1.5, 8.0]$ | Verified |

---

## 4. Corrective Actions Implemented
1. **Recalibrated Trajectory Parameters**: Increased base push-in $t_z = -0.22$, lateral drift $t_x = 0.025$, and vertical rise $t_y = -0.015$.
2. **Flow Magnitude Percentiles**: Upgraded optical flow measurement to evaluate flow magnitude percentiles ($p_{50}, p_{90}$) rather than signed mean vector norms.
3. **P0 Debug Tracing Pipeline**: Added `export_p0_raster_debug_trace` (`output/debug/raster_trace.json`, `depth_distribution.json`).
4. **P0 Frame Difference Artifacts**: Added `generate_p0_frame_difference_artifacts` (`frame_difference_report.json`, `frame_diff_f00_f99.png`, `frame_overlay_f00_f99.png`, `motion_heatmap.png`).
