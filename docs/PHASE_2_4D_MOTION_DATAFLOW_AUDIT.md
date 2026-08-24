# Phase 2.4D — Motion Data Flow Audit & Line-by-Line Forensic Inspection

## 1. Executive Summary
This document provides a line-by-line forensic audit of the motion data flow in our First-Principles Cinematic 2.5D Parallax Renderer (`v0_pipeline.py` and `spatial_intelligence/`).

The audit verifies that all novel view camera synthesis operations derive strictly from 3D pinhole perspective projection ($P' = R P + t$), with zero ungrounded arbitrary layer translations.

---

## 2. Line-by-Line Pipeline Data Flow Trace

1. **Motion Preset Selection:**
   - Input preset string (`SUBTLE_PUSH_IN`, `SLOW_DOLLY_LEFT`, `SLOW_DOLLY_RIGHT`, `VERTICAL_DRIFT`, `DIAGONAL_DOLLY`, `PARALLAX_PUSH`, `CINEMATIC_ORBIT`, `STATIC`).
2. **Camera Trajectory Generation (`generate_c1_smooth_trajectory`, line 3370):**
   - Evaluates quintic smoothstep $s_{\text{quintic}}(t) = 6t^5 - 15t^4 + 10t^3$ for $t \in [0, 1]$.
   - Generates 3D translations $[T_x(t), T_y(t), T_z(t)]$ and rotations $[\text{pitch}(t), \text{yaw}(t), \text{roll}(t)]$.
3. **Closed-Loop Safety Motion Planning (`plan_safe_motion_trajectory`, line 3246):**
   - Evaluates max disparity across trajectory frames on floor-clamped safety depth representation ($\text{safety\_depth} = \max(1.0, Z)$).
   - Iteratively scales magnitude $M$ to satisfy resolution-normalized safety ceilings ($3.0\% - 10.0\%$ image dimension).
4. **Camera Intrinsics Derivation (`derive_camera_intrinsics`, line 3461):**
   - Sets $f_x = f_y = \max(W, H)$, $c_x = W / 2$, $c_y = H / 2$.
5. **Backprojection (`back_project_points`, line 3472):**
   - Backprojects 2D grid coordinates $(u, v)$ and depth $Z$ to 3D point $P = [X, Y, Z]^T$:
     $$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z = Z$$
6. **3D Rigid Camera Transformation (`transform_3d_points`, line 3713):**
   - Transforms 3D points by rotation $R$ and translation $t$:
     $$P' = R \cdot P + t$$
7. **Perspective Projection (`project_3d_points`, line 3718):**
   - Projects transformed 3D points $P' = [X', Y', Z']^T$ back to 2D image coordinates $(u', v')$:
     $$u' = f_x \cdot \frac{X'}{Z'} + c_x, \quad v' = f_y \cdot \frac{Y'}{Z'} + c_y$$
8. **Forward Subpixel Splatting & Z-Buffering (`splat_layer`, line 3568):**
   - Bilinear weight distribution across $2 \times 2$ pixel neighborhood ($\sum w_i = 1.0$).
   - Deterministic Z-buffer ownership reset (`accum_col[v, u] = 0, accum_w[v, u] = 0`) whenever $Z_{\text{new}} < Z_{\text{buf}} - 0.001$.
9. **Disocclusion Handling & Compositing (`render_single_frame_forward_splatting`, line 3528):**
   - Pre-rendered background plate inpainting (`cv2.INPAINT_TELEA`) fills disocclusion holes on moving frames.
   - Foreground layer strictly overlays background layer where valid foreground splats exist ($fg\_w > 0.05$).
10. **Raster Optical Flow & Quality Diagnostics (`compute_perceptual_motion_score`, line 2618):**
    - Tracks peak optical flow vectors $u_{\text{peak}}, v_{\text{peak}}$ across evaluation keyframes.
    - Evaluates 3-tier diagnostic report (`ThreeTierDiagnosticReport`).

---

## 3. Zero-Motion Identity Verification

* **Test Conditions:** $T_x = T_y = T_z = 0.0$, $R = I_{3 \times 3}$.
* **Measured Metrics:**
  - Mean Absolute Error (MAE): `0.000000`
  - Root Mean Square Error (RMSE): `0.000000`
  - Max Pixel Difference: `0.0`
  - Changed Pixel Percentage: `0.00%`

* **Conclusion:** Zero-motion identity holds perfectly with zero rasterization error or unexplained pixel displacement.
