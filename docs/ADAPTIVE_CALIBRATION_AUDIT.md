# Adaptive High-Motion Calibration Audit Report

**Date:** August 15, 2026
**Status:** COMPLETE — PHASE 1 AUDIT
**Author:** First-Principles Cinematic 2.5D Parallax Renderer (V0) Pipeline

---

## Executive Summary
This audit inspects the camera trajectory, motion classification, and validation subsystems in `v0_pipeline.py` and `spatial_intelligence/perceptual_motion.py`. It identifies why the $1536 \times 1024$ 48-frame HIGH Cinematic Push-In run achieved strong internal metrics (depth parallax score = 0.813, environmental motion score = 0.859) yet resulted in `motion_classification = WEAK` and `motion_good = false`.

---

## Audited Codebase Locations

### 1. Camera Trajectory Generation
- **Function:** `generate_c1_smooth_trajectory(style, magnitude_scale, num_frames, is_loop)`
- **File:** `v0_pipeline.py` (Line ~3278)
- **Role:** Computes SE(3) camera poses ($T_x, T_y, T_z$, Pitch, Yaw, Roll) across $t \in [0, 1]$ using C1 quintic smoothstep easing $s(t) = 6t^5 - 15t^4 + 10t^3$.
- **Current Behavior:** Applies static base translations (e.g. $t_z = -0.45 \times \text{magnitude\_scale}$) without checking achieved image-space displacement.

### 2. Closed-Loop Safety Planning
- **Function:** `plan_safe_motion_trajectory(style, strength, width, height, ...)`
- **File:** `v0_pipeline.py` (Line ~3218)
- **Role:** Evaluates trajectory candidates against resolution disparity ceilings ($0.03\times, 0.06\times, 0.10\times \text{dim\_ref}$) over $Z \ge 1.0$ clipped depth bounds.
- **Current Limitation:** Bounds maximum disparity to prevent distortion, but does not enforce a floor on achieved image-space motion to guarantee preset targets are reached.

### 3. Motion Classification & Visibility
- **Function:** `classify_motion_visibility(subject_disp_px, bg_disp_px, relative_disp_px, scale_change_ratio, ...)`
- **File:** `v0_pipeline.py` (Line ~2513)
- **Role:** Classifies motion into `NEGLIGIBLE`, `SUBTLE`, `VISIBLE`, `CINEMATIC`, `EXCESSIVE`, `UNSAFE`, or `WEAK`.
- **Current Bug:** Hardcoded fixed thresholds (`bg_disp_px < 15.0 and rel_bg_sub < 6.0` $\to$ `WEAK` for HIGH preset) classify motion as `WEAK` if background displacement is $13.51\text{px}$, ignoring that $13.51\text{px}$ on a $1536 \times 1024$ image represents $0.88\%$ of image width with foreground displacement reaching $29.22\text{px}$.

### 4. Perceptual Motion Score & Gate Evaluation
- **Function:** `compute_perceptual_motion_score(...)`
- **File:** `v0_pipeline.py` (Line ~2617)
- **Role:** Evaluates global raster displacement, subject scale growth, layer optical flow, artifact ratio, and disocclusion ratio across rendered keyframe pairs (F00 to F_end).
- **Current Output:** Fails `motion_good = false` if `motion_visibility_class` returns `WEAK`.

---

## Key Findings & Calibration Strategy
1. **Separation of Intent vs Achievement:** The pipeline must explicitly record `requested_amplitude` (e.g. HIGH) and `achieved_amplitude` (e.g. MEDIUM or HIGH) based on normalized image-space displacement metrics ($\text{disp\_px} / \max(W, H)$).
2. **Adaptive Calibration Loop:** `plan_safe_motion_trajectory()` must iteratively adjust candidate trajectory amplitude ($T_z$ scale) using image-space feedback until the requested preset target is achieved or bounded safety limits (disocclusion, artifact ratio, subject scale growth $< 10\%$) are met.
3. **Resolution Normalization:** Thresholds must evaluate normalized image-space displacement ($\text{disp\_px} / \max(W, H)$) rather than unscaled pixel counts, guaranteeing consistent behavior across $320 \times 320$, $640 \times 640$, $1024 \times 1024$, and $1536 \times 1024$.
