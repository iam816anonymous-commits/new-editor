# Phase 2.4B Forensic Audit Report — Surface-Coherent Parallax & Pixel Displacement Fix

**Date:** August 15, 2026
**Status:** COMPLETE — FORENSIC AUDIT
**Author:** First-Principles Cinematic 2.5D Parallax Renderer (V0) Pipeline

---

## Executive Summary
This forensic audit investigates the root causes of visual texture swimming, pixel jitter, and surface displacement incoherence during camera motion. While previous phases achieved numerical optical flow targets, unregularized monocular depth noise and subpixel splatting collisions can cause adjacent pixels on the same physical surface to shift slightly differently.

---

## Forensic Audit Questions (1 through 10)

### 1. Can depth noise create different motion for adjacent pixels on the same surface?
- **Finding:** YES. In `splat_layer()` (`v0_pipeline.py`), back-projected 3D coordinates $P = [(u-c_x)Z/f_x, (v-c_y)Z/f_y, Z]^T$ undergo $SE(3)$ transformation. Unregularized depth noise (e.g. $Z=2.10$ vs $Z=2.25$ on a flat torso) causes different perspective divide shifts $\Delta u$, producing texture swimming.
- **Location:** `v0_pipeline.py` (Line ~3565).

### 2. Can RGB/depth edge misalignment create false geometry?
- **Finding:** YES. Monocular depth boundaries from Depth Anything V2 Small often bleed 3–5 pixels beyond RGB object contours.
- **Location:** `edge_aware_depth_refinement()` (`v0_pipeline.py`).

### 3. Can foreground depth bleed into background?
- **Finding:** YES, if bilateral filtering parameters ($\sigma_{\text{color}}=75.0, \sigma_{\text{space}}=75.0$) over-smooth depth near un-validated edges.

### 4. Can background depth bleed into foreground?
- **Finding:** YES, when background depth extrapolation propagates high $Z$ values into soft subject edge feathering zones.

### 5. Can splatting create duplicated energy?
- **Finding:** YES. In forward subpixel splatting, 1 source pixel distributes bilinear weights to 4 subpixel grid cells. If weights accumulate without Z-buffer ownership resets, intensity blooming occurs.
- **Location:** `splat_layer()` (`v0_pipeline.py`).

### 6. Can splatting create holes?
- **Finding:** YES. Camera translation $T_x, T_z$ stretches points apart ($\Delta u > 1.0$).
- **Resolution:** Telea Fast Marching inpainting fills subpixel gaps on target keyframe grids.

### 7. Can multiple source pixels compete incorrectly?
- **Finding:** YES, if the Z-buffer precision tolerance ($\epsilon = 0.001$) allows occluded background points to blend into foreground subpixels.

### 8. Can the current renderer deform a rigid-looking surface?
- **Finding:** YES, if connected regions in `ParallaxRegion` do not share unified motion coupling or surface regularization.

### 9. Can reconstructed pixels differ between frames?
- **Finding:** YES, if inpainting is re-computed on every frame rather than using a temporally persistent background plate (`background_plate.png`).

### 10. Are current visual-quality metrics measuring actual geometric error or merely motion magnitude?
- **Finding:** Previously, optical flow magnitude variance was evaluated.
- **Resolution:** Introduce 3D expected geometric flow $d_{\text{expected}}$ and surface residual flow error $e_{\text{residual}} = \|d_{\text{observed}} - d_{\text{expected}}\|$ per `ParallaxRegion`.
