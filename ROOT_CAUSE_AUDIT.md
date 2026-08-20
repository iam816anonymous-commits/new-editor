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

## 2. Identified Root Causes of Visual Motion Loss

### Root Cause 1: Perspective Coordinate Scale & Intrinsics Mapping
- **Issue**: Standard image intrinsics ($f_x = f_y = \max(W, H) = 1536$) set focal length equal to image width. For a background pixel at depth $Z = 5.0$, a Z push-in $\Delta Z = -0.10$ creates a scale expansion ratio of $\frac{5.0}{5.0 - 0.10} = 1.0204$ (+2% expansion).
- **Effect**: On a $1536 \times 1024$ image, a center-region background pixel shifts by only $\approx 4.7\text{px}$. To a human watching at 24 FPS, a 4.7px shift over 48–100 frames translates to $<0.1\text{px/frame}$, appearing completely static.

### Root Cause 2: Subject Anchoring Over-Constraining Environmental Travel
- **Issue**: To preserve primary subject dignity, the primary subject multiplier was constrained to $0.15\times - 0.35\times$. However, because $t_z$ was small (0.10), the base background multiplier ($1.00\times$) only received $t_z = -0.10$, while the subject received $t_z = -0.035$.
- **Effect**: The relative background vs subject disparity was under $3\text{px}$, causing both the subject AND the environment to look static.

### Root Cause 3: Open-Loop Motion Planning
- **Issue**: The trajectory generator produced camera poses open-loop without measuring the resulting raster pixel displacement before rendering full frames.
- **Effect**: If scene depth distribution was compressed (e.g. depth range [2.0, 3.5]), the fixed camera pose produced drastically under-threshold raster displacement without automatic trajectory correction.

### Root Cause 4: Optical Flow Vector Cancellation in Signed Measurements
- **Issue**: Farneback optical flow computes signed velocity vectors $(u, v)$. Averaging $u$ and $v$ over symmetric layer regions caused left-moving and right-moving vectors to cancel out to near-zero ($0.02\text{px}$).
- **Fix Required**: Measure optical flow using flow magnitude percentiles ($p_{50}, p_{90}$) and directional coherence rather than raw signed mean vectors.

---

## 3. Corrective Action Plan for Phase 2.3B/P0
1. **Recalibrate Trajectory Translation Scale**: Increase base push-in translation to $t_z = -0.22$, lateral drift $t_x = 0.025$, and vertical rise $t_y = -0.015$.
2. **Implement Closed-Loop Trajectory Calibration**: `PLAN -> PROXY RENDER -> MEASURE RASTER -> CORRECT TRAJECTORY` to guarantee target raster displacement in pixel space.
3. **Upgrade Optical Flow Measurement**: Use Farneback flow magnitude percentiles and directional variance across independent layer masks.
4. **Separate Camera Intent from Raster Results**: Report `camera_intent`, `raster_result`, and `perceptual_result` independently in `camera_vs_raster_motion.png` and `validation_summary.json`.
