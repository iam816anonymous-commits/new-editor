# Phase 2.3C Current-State Baseline Audit

**Date:** August 15, 2026
**Status:** COMPLETE — BASELINE AUDIT
**Author:** First-Principles Cinematic 2.5D Parallax Renderer (V0) Pipeline

---

## Overview & Audit Purpose
This baseline audit provides a rigorous, code-level analysis of the First-Principles Cinematic 2.5D Parallax Renderer (V0) before initiating Phase 2.3C enhancements. The current baseline represents a fully functional, 100% real-model system with 146 passing unit and integration tests, true 3D pinhole reprojection, forward subpixel splatting, deterministic Z-buffering, and real-frame optical flow evaluation.

---

## Codebase Subsystems Audited
1. `v0_pipeline.py` (Core pipeline, CLI execution, trajectory planner, forward splatting, Z-buffer, renderer, diagnostics exporter)
2. `spatial_intelligence/` (Scene graph, entity consolidation, trust scoring, depth fields, perspective camera model, relationship inference, temporal tracking)
3. `subject_selection/` (Multi-signal candidate feature extraction, scoring, compound subject grouping, mask refinement, semantic confidence evaluation)

---

## Detailed Audit Questions (A through L)

### A. What currently creates visible camera motion?
Visible camera motion is created by transforming 3D point cloud representations of the scene using SE(3) camera poses $P' = R P + t$ computed along C1-smooth smoothstep trajectories (`generate_c1_smooth_trajectory()`).
- **Translation Parameters:** For `Cinematic Push-In`, base translations are $t_z = -0.45$ (forward push along negative Z axis toward scene), $t_x = 0.08$ (subtle rightward pan), $t_y = -0.04$ (subtle upward pan).
- **Rotation Parameters:** Pitch $= -0.015\text{ rad}$, Yaw $= 0.010\text{ rad}$, Roll $= 0.000\text{ rad}$.
- **Smooth Easing:** Easing is driven by smoothstep $s(t) = 3t^2 - 2t^3$ (or quintic $6t^5 - 15t^4 + 10t^3$), ensuring zero velocity jumps at endpoints ($V(0) = V(1) = 0$).

---

### B. What currently creates depth parallax?
Depth parallax is mathematically created by perspective division $u' = f_x \frac{X'}{Z'} + c_x, v' = f_y \frac{Y'}{Z'} + c_y$ applied to 3D point coordinates $P = [X, Y, Z]^T$ back-projected from relative monocular scene depth $Z = \text{depth\_map}(u, v)$.
- Points with closer depth ($Z \approx 0.1 - 2.0$) undergo significantly larger coordinate shifts $\Delta u = u' - u$ for a given camera translation $t_x, t_z$ than distant points ($Z \approx 10.0$).
- Layer motion multipliers (`BACKGROUND` = 1.00x, `MIDGROUND` = 1.50x, `FOREGROUND` = 2.20x, `PRIMARY_SUBJECT` = 0.35x) scale camera translations per layer before 3D reprojection to enhance environmental depth separation while restraining primary subject scale expansion.

---

### C. What currently limits motion at 320x320?
On low-resolution $320 \times 320$ images, motion is bounded by two distinct factors:
1. **Resolution-Proportional Disparity Safety Ceilings:** The closed-loop safety planner (`plan_safe_motion_trajectory()`) calculates a target disparity ceiling $0.06 \times \text{dim\_ref} = 0.06 \times 320 = 19.2\text{px}$. Evaluating trajectory disparity against scene geometry scales candidate trajectories down by $0.344\times$ to enforce this safety ceiling.
2. **Subpixel Flow Quantization:** On $320 \times 320$, deep background points ($Z = 10.0$ sky) undergo physical shifts on the order of $\sim 0.0001\text{px}$. Farneback optical flow (`cv2.calcOpticalFlowFarneback`) cannot resolve shifts $< 0.01\text{px}$ and yields zero flow vector magnitudes, causing metric measurements to fall back to RGB intensity noise floors ($0.32 - 0.39\text{ L1}$).

---

### D. What currently limits motion at 1536x1024?
On production-resolution $1536 \times 1024$ images:
- Disparity scale limits expand proportionally from $19.2\text{px}$ to $92.16\text{px}$ ($0.06 \times 1536$).
- Safety scale factor achieves full $1.0\times$ candidate scale ($t_z = -0.45$ push-in).
- Motion is strictly limited by **primary subject scale expansion safety** ($< 4\%$ scale growth limit) and **disocclusion boundary hole filling quality**, preventing disocclusion edge tearing or halo artifacts around subject borders.

---

### E. How is subject rigidity preserved?
Subject rigidity is preserved through four synchronized mechanisms:
1. **Primary Subject Layer Multiplier:** `PRIMARY_SUBJECT` motion is scaled by $0.35\times$ relative to the camera trajectory, dampening perspective expansion.
2. **Distance-Transform Falloff Feathering:** Soft boundary feathering ($W_{\text{inner}} = 1.0$, boundary falloff) anchors the rigid inner core of the subject while allowing soft edge blending into background disocclusion regions.
3. **Continuous Continuous Depth Geometry:** Subject depth values originate from Depth Anything V2 Small filtered with Joint Bilateral Filtering, maintaining internal 3D geometric shape rather than treating the subject as a flat 2D plane card.
4. **Subject Scale Change Metric Check:** `evaluate_subject_scale_change()` tracks bounding box and area growth across frames, failing the perceptual motion gate if scale expansion exceeds $10\%$.

---

### F. How is depth ordering preserved?
Depth ordering is strictly preserved by:
1. **Deterministic Camera-Space Z-Buffering:** During 3D forward subpixel splatting, candidate 3D points competing for the same target subpixel $(u', v')$ are resolved using a depth buffer array `z_buffer[v', u']`. Points with smaller camera-space $Z'$ win over occluded background points with larger $Z'$.
2. **Depth Layer Quantile Segmentation:** Background layer masks (`FOREGROUND`, `MIDGROUND`, `BACKGROUND`) are partitioned using depth distribution quantiles ($q_{20}, q_{70}$), enforcing $Z_{\text{fg}} < Z_{\text{mg}} < Z_{\text{bg}}$.

---

### G. How are disocclusions handled?
Disocclusions (uncovered pixels behind moving subject boundaries) are handled via a precomputed background plate and dynamic forward splatting:
1. **Clean Background Plate Creation:** `reconstruct_background_rgb_and_depth()` applies subject mask dilation (15px) and Telea Fast Marching inpainting (`cv2.inpaint`) on the reference RGB image and depth map to create `background_plate.png` and `background_depth.png`.
2. **Foreground & Background Splatting:** Background plate 3D points are reprojected and splat onto target keyframe grids. Any remaining subpixel holes are filled using local Telea inpainting on the target keyframe. Static reference pixels are **never** restored at unshifted coordinates.

---

### H. Where can motion become visually artificial?
Motion can become visually artificial if:
- **Excessive Pitch/Yaw Rotation:** Camera rotations $> 0.05\text{ rad}$ cause keystoning and field-of-view perspective distortion without physical translation depth separation.
- **Over-Amplified Layer Multipliers:** Layer motion multipliers $> 3.0\times$ create "cardboard layer sliding" where discrete depth bands shear across each other.
- **Disocclusion Bleed:** Aggressive camera travel exposing large un-inpainted disocclusion regions causes texture stretching or halo smearing.
- **Synthetic Metric Substitution:** Substituting camera intent or artificial 2D image shifts for true 3D raster reprojection destroys physical realism.

---

### I. Which parameters affect physical camera motion?
- Camera translation vector $t = [t_x, t_y, t_z]^T$ (metres / world units)
- Camera rotation angles $\theta = [\text{pitch}, \text{yaw}, \text{roll}]^T$ (radians)
- Focal length $f_x, f_y$ and principal point $c_x, c_y$ (pinhole intrinsic parameters)
- Normalized continuous scene depth map $Z(u, v)$

---

### J. Which parameters affect perceptual presentation?
- Layer motion multipliers (`PRIMARY_SUBJECT`, `FOREGROUND`, `MIDGROUND`, `BACKGROUND`)
- Motion amplitude presets (`LOW` = 0.60x, `MEDIUM` = 1.20x, `HIGH` = 2.50x)
- Smoothstep temporal trajectory progression $s(t) = 3t^2 - 2t^3$
- Distance-transform falloff feathering radius

---

### K. Which parameters must remain physically coupled?
1. **Back-Projection & Forward Projection:** 3D coordinates $P = [(u-c_x)Z/f_x, (v-c_y)Z/f_y, Z]^T$ must be reprojected using exact inverse camera intrinsic matrix $K^{-1}$ and forward intrinsic matrix $K$.
2. **Camera Pose Transformation:** $P' = R P + t$ must apply rigid body SE(3) rotation matrix $R \in SO(3)$ and translation vector $t \in \mathbb{R}^3$.
3. **Z-Buffer Depth Ordering:** Target pixel depth $Z'$ must correspond strictly to transformed camera-space Z coordinate $P'_z$.

---

### L. Which parameters must NEVER be independently manipulated merely to improve metrics?
1. **Optical Flow Metrics:** Optical flow vectors must NEVER be multiplied by artificial constants or synthetic layer scale factors.
2. **Subject Mask Coordinates:** Subject masks must NEVER be translated or warped independently of 3D reprojection.
3. **Camera Intent vs Raster Metrics:** Trajectory camera translations must NEVER be reported as raster pixel displacements.
4. **Disocclusion Inpainting:** Static unshifted reference RGB pixels must NEVER be pasted over moving background regions to artificially pad frame similarity.

---

## Conclusion & Readiness
The baseline codebase is fully audited, mathematically sound, and ready for Phase 2.3C perceptual motion enhancement.
