# Phase 2.3C Technical Reference Study — Perceptual Cinematic Motion & Depth Coherence

**Date:** August 15, 2026
**Status:** COMPLETE — TECHNICAL STUDY
**Author:** First-Principles Cinematic 2.5D Parallax Renderer (V0) Pipeline

---

## Executive Summary
This technical reference study explores literature from Depth-Image-Based Rendering (DIBR), view synthesis, optical flow analysis, cinematographic camera motion, depth-aware image deformation, and layered depth image (LDI) representation. Its purpose is to establish mathematically sound conceptual adaptations for Phase 2.3C without copying external source code or introducing unphysical shortcuts.

---

## 1. Classical Depth-Image-Based Rendering (DIBR) & View Synthesis

### 1. Technique
Classical DIBR transforms a single 2D RGB image $I$ and associated depth map $D$ into novel camera viewpoints $I'$ using pinhole back-projection, rigid 3D pose transformation, and forward projection.

### 2. Mathematical Principle
Given image space pixel coordinates $(u, v)$ and normalized inverse/direct depth $Z(u, v)$:
$$P = \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \begin{bmatrix} (u - c_x) \frac{Z}{f_x} \\ (v - c_y) \frac{Z}{f_y} \\ Z \end{bmatrix}$$
Transforming by camera extrinsic rotation matrix $R \in SO(3)$ and translation vector $t \in \mathbb{R}^3$:
$$P' = \begin{bmatrix} X' \\ Y' \\ Z' \end{bmatrix} = R P + t$$
Projecting back onto the normalized camera sensor grid $(u', v')$:
$$u' = f_x \frac{X'}{Z'} + c_x, \quad v' = f_y \frac{Y'}{Z'} + c_y$$

### 3. Relevance to Renderer
Forms the core geometric transformation engine in `v0_pipeline.py`. Correct DIBR guarantees mathematically true 3D perspective shifts rather than 2D planar translations.

### 4. What We Adapt Conceptually
- Forward subpixel splatting with bilinear accumulation weights.
- SE(3) camera pose formulation ($R, t$).
- Continuous depth field representation rather than discrete cardboard cards.

### 5. What We Should NOT Copy
- Simple backward warping (e.g. `cv2.remap` or `cv2.warpAffine`), which causes severe stretching artifacts at occlusion boundaries and destroys Z-buffer visibility ordering.

### 6. Expected Perceptual Quality Effect
High visual realism; natural motion parallax where foreground objects move faster than background objects.

### 7. Computational Cost
Vectorized NumPy/PyTorch GPU/CPU matrix multiplication: $\mathcal{O}(N)$ where $N = W \times H$ pixels ($\sim 15 - 30\text{ ms}$ per frame).

### 8. Risk of Introducing Artifacts
Disocclusion holes (uncovered background pixels) and subpixel grid quantization gaps if splatting weights or inpainting are uncalibrated.

---

## 2. Cinematographic Push-In & Camera Trajectory Perception

### 1. Technique
Cinematographic dolly push-in moves the virtual camera monotonically forward toward the subject along the optical Z-axis ($t_z < 0$), combined with subtle compensatory panning ($t_x, t_y$) and minor pitch/yaw rotations.

### 2. Mathematical Principle
In a push-in trajectory, screen disparity expansion $\Delta u$ is governed by:
$$\Delta u = (u - c_x) \left( \frac{Z}{Z + t_z} - 1 \right) = (u - c_x) \frac{-t_z}{Z + t_z}$$
For close objects ($Z \approx 1.0$), $-t_z / (Z + t_z)$ is large, producing strong expansion. For background ($Z \approx 10.0$), $-t_z / (Z + t_z)$ is small, producing minimal expansion.

### 3. Relevance to Renderer
Governs `generate_c1_smooth_trajectory()` for `Cinematic Push-In`. It creates the desired psychological feeling: "The viewer feels the camera is physically moving into the 3D space."

### 4. What We Adapt Conceptually
- Smoothstep temporal progression $s(t) = 3t^2 - 2t^3$ or quintic $s(t) = 6t^5 - 15t^4 + 10t^3$.
- Coupled translation and rotation ($t_z, t_x, t_y, \text{pitch}, \text{yaw}$) to maintain focal tracking on the subject.

### 5. What We Should NOT Copy
- Unconstrained $t_z$ magnitudes that cause perspective divide explosion ($Z + t_z \to 0$) or extreme subject scale growth ($> 10\%$).

### 6. Expected Perceptual Quality Effect
Immersive, cinematic depth travel that draws the viewer's attention into the scene while maintaining subject focal anchor.

### 7. Computational Cost
Negligible ($\mathcal{O}(K)$ where $K$ is frame count, $\sim 0.1\text{ ms}$).

### 8. Risk of Introducing Artifacts
If $t_z$ is too large, subject scale growth causes "breathing" or rubber-sheet distortion.

---

## 3. Optical Flow & Farneback Motion Analysis

### 1. Technique
Farneback dense optical flow estimates two-dimensional displacement field vectors $\vec{v}(u, v) = (v_x, v_y)$ between consecutive or keyframe image pairs $(I_0, I_k)$.

### 2. Mathematical Principle
Solves the local polynomial expansion problem $f(x) \sim x^T A x + b^T x + c$ under brightness constancy assumptions $I(x, y, t) = I(x + v_x, y + v_y, t + \delta t)$.
Local motion vector magnitude:
$$M(u, v) = \|\vec{v}(u, v)\|_2 = \sqrt{v_x^2 + v_y^2}$$

### 3. Relevance to Renderer
Used in `compute_perceptual_motion_score()` and `spatial_intelligence/perceptual_motion.py` to evaluate actual rendered frame displacements across depth quantiles ($Z_{\text{fg}}, Z_{\text{mg}}, Z_{\text{bg}}$).

### 4. What We Adapt Conceptually
- Measuring motion from actual rendered keyframes (F00 to F47/F99).
- Quantile-based layer evaluation ($p_{90}$, median) to handle localized occlusions and edge noise.

### 5. What We Should NOT Copy
- Multiplying optical flow measurements by artificial scale constants.
- Relying on optical flow alone at subpixel motion scales ($< 0.01\text{px}$) where quantization noise dominates.

### 6. Expected Perceptual Quality Effect
Provides objective, mathematical ground-truth verification of physical raster displacement.

### 7. Computational Cost
$\mathcal{O}(W \times H)$ per frame pair ($\sim 5 - 15\text{ ms}$ per keyframe pair using OpenCV).

### 8. Risk of Introducing Artifacts
At subpixel levels ($< 0.01\text{px}$), Farneback flow returns 0.0, requiring dual evaluation with L1 RGB intensity difference.

---

## 4. Layered Depth Images (LDI) & Disocclusion Inpainting

### 1. Technique
Layered Depth Images represent background surfaces hidden behind primary foreground subjects, enabling clean view synthesis during camera travel without revealing empty holes.

### 2. Mathematical Principle
Background RGB $I_{\text{bg}}(u, v)$ and background depth $D_{\text{bg}}(u, v)$ are estimated via conservative subject mask dilation $\mathcal{M}_{\text{dilated}} = \mathcal{M} \oplus K_{\text{radius}}$ and Fast Marching Method (Telea inpainting) solving the eikonal equation:
$$|\nabla T| = 1 \quad \text{on } \partial \Omega$$

### 3. Relevance to Renderer
Implemented in `reconstruct_background_rgb_and_depth()` to generate clean background plates (`background_plate.png` and `background_depth.png`).

### 4. What We Adapt Conceptually
- Dilation radius scaled to expected boundary disparity.
- Independent background plate reprojection during forward splatting.

### 5. What We Should NOT Copy
- Pasting unshifted static reference RGB pixels back over moving background regions, which cancels camera movement.

### 6. Expected Perceptual Quality Effect
Seamless disocclusion revealing clean, natural background textures behind moving subjects.

### 7. Computational Cost
Inpainting execution: $\mathcal{O}(M_{\text{hole}} \log M_{\text{hole}})$ ($\sim 100 - 300\text{ ms}$ during precomputation).

### 8. Risk of Introducing Artifacts
Over-dilation causes texture blurring or halo smearing if background textures are highly structured.

---

## Summary of Adaptations for Phase 2.3C

1. **Pure Physical Reprojection:** All motion is generated strictly via 3D SE(3) pose transformation and pinhole perspective division.
2. **Subject Rigidity Guarantee:** Primary subject motion is restrained ($0.35\times$ layer multiplier, distance-transform falloff feathering) to keep area growth $< 4\%$.
3. **Multi-Layer Depth Parallax:** Background layers ($Z_{\text{fg}} < Z_{\text{mg}} < Z_{\text{bg}}$) receive depth-weighted translation multipliers ($2.20\times, 1.50\times, 1.00\times$) to enforce $\text{Foreground} > \text{Midground} > \text{Background}$ displacement ordering.
4. **Independent Raster Measurement:** All metrics are measured from rendered PNG keyframes via Farneback optical flow and L1 RGB difference, completely independent of camera intent parameters.
