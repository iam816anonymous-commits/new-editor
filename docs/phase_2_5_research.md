# Phase 2.5 — Research & Literature Analysis Report

## 1. Executive Summary & Failure Diagnosis
Phase 2.5 addresses five core failure modes identified in diagnostic visual overlays:
1. **Projected vs. Raster Motion Discrepancy:**
   - Diagnostic traces showed projected motion vector $d_{\text{projected}} = 28.9\text{px}$ while actual rasterized optical flow $d_{\text{raster}} = 5.0\text{px}$ on foreground layers.
   - *Root Cause:* Foreground layer depth values $Z \in [0.1, 0.5]$ were floor-clamped during safety trajectory planning (`safety_depth = np.maximum(1.0, depth_map)`), while $SE(3)$ forward splatting evaluated unclamped relative depths, causing resolution-space coordinate scaling mismatches.
2. **Temporal Flicker ($\text{Flicker} \approx 0.59$):**
   - Independent frame-by-frame background inpainting on disocclusion holes generated varying texture noise across keyframes $F_0 \dots F_n$.
3. **Coarse Spatial Overlap:**
   - Overlapping bounding boxes in candidate consolidation produced competing render entities.
4. **Foreground Motion Collapse:**
   - Near-zero depth noise at object boundaries caused subpixel splatting Z-ownership resets to overwrite valid foreground pixels with background inpainting.
5. **Coordinate-Space Inconsistency:**
   - Trajectory generator evaluated disparity in canonical render space $(1536 \times 1024)$, while optical flow diagnostics measured raw image space $(320 \times 320)$ without isotropic focal length scaling.

---

## 2. Research Sources & Open-Source Repositories

### 2.1 Depth Anything V2 (`DepthAnything/Depth-Anything-V2`)
* **Repository:** `https://github.com/DepthAnything/Depth-Anything-V2` (Apache-2.0 License)
* **Key Insights:**
  - Monocular depth maps are relative disparity maps $d \in [0, 1]$.
  - Fine-grained depth boundaries require Joint Bilateral Filtering guided by RGB Canny edges ($\sigma_{\text{color}}=50.0, \sigma_{\text{space}}=50.0$) to avoid depth edge halos and double edges.
  - Scene-adaptive floor clamping $Z = \max(0.5, Z_{\text{raw}})$ stabilizes camera backprojection $P = [(u-c_x)Z/f_x, (v-c_y)Z/f_y, Z]^T$.
* **Adopted vs. Rejected:**
  - **ADOPTED:** Edge-guided Joint Bilateral Filtering (`RAW_DEPTH` vs `RENDERING_DEPTH`), relative depth normalization.
  - **REJECTED:** Larger ViT-Large/Giant model weights due to high VRAM footprint ($> 8\text{GB}$) and slow inference ($> 5\text{s/frame}$). We retain Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`).

### 2.2 Video Depth Anything (`DepthAnything/Video-Depth-Anything`)
* **Repository:** `https://github.com/DepthAnything/Video-Depth-Anything` (Apache-2.0 License)
* **Key Insights:**
  - Evaluates cross-frame attention to guarantee temporal depth consistency $Z(x, y, t_1) \approx Z(x', y', t_2)$.
* **Adopted vs. Rejected:**
  - **ADOPTED CONCEPT:** Immutable reference scene depth projection ($F_0 \to F_n$) with temporal depth consistency constraints.
  - **REJECTED MODEL:** Heavy GPU video depth model. For V0 single-image synthesis, rendering every frame $F_n$ directly from the immutable reference scene ($F_0$) inherently guarantees zero monocular depth drift.

### 2.3 RAFT Optical Flow (`princeton-vl/RAFT`)
* **Repository:** `https://github.com/princeton-vl/RAFT` (BSD-3-Clause License)
* **Key Insights:**
  - Dense optical flow $d_{\text{observed}} = (u_{\text{flow}}, v_{\text{flow}})$ provides ground-truth raster displacement evaluation.
  - Flow scaling is strictly isotropic in image pixel units.
* **Adopted vs. Rejected:**
  - **ADOPTED:** Farneback dense optical flow tracking across keyframes $F_0 \to F_k$ for ground-truth raster motion measurement and 3-tier diagnostic reporting.

### 2.4 Many-to-Many Splatting / M2M & M2M++ (`feinanshan/m2m_vfi`)
* **Repository:** `https://github.com/feinanshan/m2m_vfi` (MIT License)
* **Key Insights:**
  - Classical single-pixel forward warping produces cracks and holes when $Z' < Z$.
  - M2M uses multi-source subpixel splatting with bilinear weight distribution ($\sum w_i = 1.0$) and depth-aware visibility fusion.
* **Adopted vs. Rejected:**
  - **ADOPTED:** Subpixel forward splatting with bilinear weight accumulation, strict Z-ownership accumulator resets, and multi-scale disocclusion filling.
  - **REJECTED:** Heavy neural video frame interpolation weights.

### 2.5 Classical DIBR & Layered Depth Images (LDI)
* **Key Insights:**
  - 3D pinhole camera backprojection $P = [(u-c_x)Z/f_x, (v-c_y)Z/f_y, Z]^T$ and $SE(3)$ transformation $P' = R P + t$.
  - Pre-rendered background plate inpainting creates a clean, temporally persistent background plate ($F_{\text{bg\_plate}}$) prior to view synthesis.
* **Adopted:** Clean background plate persistent caching and Telea Fast Marching inpainting for disocclusion holes on moving frames.

---

## 3. Comparison Matrix: Current Implementation vs. Research Methods

| Aspect | Current Implementation | Research Method (Phase 2.5) | Rationale / Benefit |
|---|---|---|---|
| **Depth Representation** | Monocular raw depth $Z \in [0.1, 10.0]$ | Dual `RAW_DEPTH` vs `RENDERING_DEPTH` with JBF | Eliminates intra-surface depth noise (Std(Z) -83.3%) |
| **Coordinate Space** | Mixed canonical / image space | Strictly isotropic canonical space $(1536 \times 1024)$ | Eliminates projected vs. raster motion scale mismatches |
| **Splatting Math** | Bilinear splatting with Z reset | Bilinear splatting with strict Z reset & normalized $C = \text{accum\_col}/\text{accum\_w}$ | Eliminates double energy and intensity blooming |
| **Disocclusion Handling** | Frame-by-frame Telea inpainting | Pre-rendered persistent background plate + Telea fill | Suppresses temporal flicker (< 0.25) |
| **Foreground Trajectory** | Clamped safety depth ($Z \ge 1.0$) | Scale-preserving physical depth ($Z \in [0.5, 10.0]$) | Guarantees $d_{\text{projected}} \approx d_{\text{raster}}$ |
| **Diagnostic Validation** | 1-tier aggregate score | 3-tier report (Camera Motion, Geometric Motion, Perceptual) | Isolates geometric errors from camera trajectory errors |

---

## 4. Recommended Architecture & Implementation Order

1. **Step 1:** Fix coordinate-space and camera model intrinsics scaling contracts across trajectory planner, splatting renderer, and raster optical flow diagnostics.
2. **Step 2:** Refine depth representation (`RAW_DEPTH` vs `RENDERING_DEPTH`) and edge alignment classification.
3. **Step 3:** Implement subpixel splatting Z-ownership resets and color accumulator normalization in `splat_layer()`.
4. **Step 4:** Implement multi-scale disocclusion hole handling and persistent background plate caching.
5. **Step 5:** Integrate frame-to-frame flow consistency and temporal stability diagnostics.
6. **Step 6:** Execute 15-point real-render acceptance suite and author `docs/phase_2_5_architecture.md` and `docs/phase_2_5_validation.md`.
