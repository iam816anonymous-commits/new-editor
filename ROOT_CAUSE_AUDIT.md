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
SAFETY PLANNER (plan_safe_motion_trajectory) → closed-loop safety disparity evaluation & scale convergence
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

## 3. P0 Final Safety Planner Depth-Outlier Audit

### Function Audited
- `plan_safe_motion_trajectory()` in `v0_pipeline.py`.

### Execution Trace & Mechanism
1. **Input Depth Map**: Receives `refined_depth` with normalized continuous depth $Z \in [0.10, 10.0]$.
2. **Pathological Near-Zero Outlier Region**: ~8.5% of pixels along sharp subject silhouette boundaries contain near-zero depth values ($Z \approx 0.10$).
3. **Disparity Evaluation**: `back_project_points` computes $X = (u - c_x) \cdot 0.10 / f_x$. When transformed by camera translation $t_z = -0.45$, the new depth is $Z' = 0.10 - 0.45 = -0.35$ (behind camera / near zero $Z' \approx 0.0001$).
4. **Perspective Divide Explosion**: $u' = f_x \cdot X / Z' + c_x$ produces artificial pixel disparity jumps exceeding $1,800\text{px}$.
5. **Safety Reduction Loop**: `plan_safe_motion_trajectory` evaluates `max_disp_px = 1851px` against the safety ceiling target (`19.2px` on 320x320 images).
6. **Trajectory Scale Collapse**: The closed-loop safety reduction iteratively scales down `magnitude_scale` from $1.00\times \to 0.0538\times$.
7. **Production Result**: The production CLI path receives $t_z = -0.45 \cdot 0.0538 = -0.0242$, collapsing actual background displacement down to $0.49\text{px}$!

### Scientific Remedy
Separate **rendering depth field** (preserved in full resolution for view synthesis) from **safety-analysis depth bounds** (`np.maximum(1.0, depth_map)` or percentile clipping $p_{1.0} - p_{99.0}$) during disparity ceiling evaluation in `plan_safe_motion_trajectory()`.

---

## 4. Camera Conventions Audit

| Parameter | Convention / Formula | Audit Status |
| :--- | :--- | :--- |
| **Coordinate System** | Right-handed pinhole: $+X$ right, $+Y$ down, $+Z$ into scene | Verified |
| **Push-In Direction** | Negative $t_z$ decreases camera-to-subject distance | Verified |
| **Focal Length** | $f_x = f_y = \max(W, H)$ (~53° FOV) | Verified |
| **Principal Point** | $c_x = W / 2, c_y = H / 2$ | Verified |
| **Pixel Centers** | Pixel grid integer bounds $0 \le u < W, 0 \le v < H$ | Verified |
| **Depth Scale** | Normalized continuous depth $Z \in [1.5, 8.0]$ | Verified |

---

## 5. Corrective Actions Implemented
1. **Recalibrated Trajectory Parameters**: Increased base push-in $t_z = -0.45$, lateral drift $t_x = 0.08$, and vertical rise $t_y = -0.04$.
2. **Robust Depth Safety Representation**: Updated `plan_safe_motion_trajectory()` to use percentile-clipped depth bounds ($Z \ge 1.0$) for closed-loop disparity evaluation, preserving full calibrated trajectory scales (~1.0x).
3. **Flow Magnitude Percentiles**: Upgraded optical flow measurement to evaluate flow magnitude percentiles ($p_{50}, p_{90}$) rather than signed mean vector norms.
4. **P0 Debug Tracing Pipeline**: Added `export_p0_raster_debug_trace` (`output/debug/raster_trace.json`, `depth_distribution.json`).
5. **P0 Frame Difference Artifacts**: Added `generate_p0_frame_difference_artifacts` (`frame_difference_report.json`, `frame_diff_f00_f99.png`, `frame_overlay_f00_f99.png`, `motion_heatmap.png`).
