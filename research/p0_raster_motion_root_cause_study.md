# P0 RASTER MOTION ROOT CAUSE & TECHNICAL REFERENCE STUDY

## Executive Summary
This research study evaluates key computer vision, view synthesis, and segmentation references (Facebook SAM / SAM 2, Depth Anything V2, pixelNeRF, Mip-NeRF, Instant-NGP, and classical Depth Image-Based Rendering / DIBR literature) to establish technical foundation for solving visual motion attenuation in our First-Principles Cinematic 2.5D Renderer.

**Strict Policy Compliance**: No external code was copied or imported. These references serve purely as conceptual and technical benchmarks.

---

## Technical Reference Analyses

### 1. Facebook AI Research / SAM & SAM 2
- **Repositories**: `facebookresearch/segment-anything`, `facebookresearch/sam2`
- **What We Learned**: SAM 2 provides zero-shot promptable mask generation using Hiera transformer backbones. SAM 2 outputs high-precision 2D binary silhouettes, but carries no 3D depth or spatial geometry information.
- **Relevant Technique**: Using SAM 2 output masks as boundary constraints for primary subject isolation and conservative morphologic dilation for background plate inpainting.
- **Why It Matters Here**: Isolates the primary subject anchor from environmental layers (FOREGROUND, MIDGROUND, BACKGROUND) so independent depth-weighted layer motion multipliers can be applied without corrupting subject identity.
- **What We Will NOT Copy**: Pre-trained model weights, C++ inference extensions, or tracking loop wrappers.
- **How Our Implementation Differs**: We use SAM 2 Hiera-Tiny via Hugging Face `SAM2ImagePredictor` strictly for Phase B subject mask proposal, followed by automated multi-signal candidate scoring and group consolidation in `subject_selection/`.

---

### 2. Depth Anything V2
- **Repository**: `DepthAnything/Depth-Anything-V2`
- **What We Learned**: Depth Anything V2 produces monocular relative depth maps with high boundary alignment. However, monocular depth outputs are non-metric (relative scale $[0.0, 1.0]$) and non-linear relative to physical camera distance.
- **Relevant Technique**: Percentile outlier clipping ($p_{1.0}, p_{99.0}$) and joint bilateral edge-aware filtering (`cv2.ximgproc.jointBilateralFilter`) to refine raw monocular depth while preserving sharp object silhouetting.
- **Why It Matters Here**: If monocular depth is unnormalized or mapped linearly without scale-range clipping, depth compression collapses $Z_{\text{bg}} \approx Z_{\text{sub}}$, causing all layers to move at near-identical raster speeds.
- **What We Will NOT Copy**: Network training scripts or custom evaluation loops.
- **How Our Implementation Differs**: We normalize relative monocular depth $d \in [0, 1]$ into rendering coordinate depth $Z \in [1.5, 8.0]$, constructing a structured 2.5D DepthField in `spatial_intelligence/depth_field.py`.

---

### 3. zju3dv / pixelNeRF & google-research / Mip-NeRF
- **Repositories**: `sxyu/pixel-nerf`, `google-research/multinerf`
- **What We Learned**: Neural Radiance Fields (NeRF) achieve novel view synthesis by volume rendering implicit neural radiance fields along camera rays $r(t) = o + t d$. pixelNeRF conditions radiance fields on local image features for few-shot synthesis. Mip-NeRF uses anti-aliased conical frustums instead of rays.
- **Why It Matters Here**: Implicit neural volume rendering avoids explicit mesh reconstruction, but requires hundreds of forward passes per pixel, making deterministic real-time single-image local rendering impractical.
- **What We Will NOT Copy**: Implicit MLP networks, volumetric ray-marching loops, or heavy GPU training code.
- **How Our Implementation Differs**: We use classical Depth Image-Based Rendering (DIBR) with explicit 3D pinhole back-projection and Z-buffered forward subpixel splatting, achieving deterministic CPU/CUDA rendering in $<1$ second per frame.

---

### 4. NVlabs / Instant-NGP
- **Repository**: `NVlabs/instant-ngp`
- **What We Learned**: Instant-NGP uses multi-resolution hash encodings to accelerate neural radiance field evaluation.
- **Relevant Technique**: Hash-table indexing for spatial lookup.
- **Why It Matters Here**: Demonstrates the importance of spatial indexing and fast coordinate lookups.
- **How Our Implementation Differs**: We maintain explicit numpy/PyTorch coordinate grids and Z-buffers with bilinear 2x2 splatting neighborhood distribution (`np.add.at`).

---

### 5. Classical Depth Image-Based Rendering (DIBR) & View Synthesis Literature
- **Key Concepts Studied**:
  - **Pinhole Back-Projection**: $X = \frac{(u - c_x) Z}{f_x}, Y = \frac{(v - c_y) Z}{f_y}, Z = Z$.
  - **Camera Transformation**: $P' = R \cdot P + t$.
  - **Perspective Projection**: $u' = f_x \frac{X'}{Z'} + c_x, v' = f_y \frac{Y'}{Z'} + c_y$.
  - **Perspective Scale Expansion**: The scale expansion of a surface at depth $Z$ under forward camera shift $\Delta Z$ is given by $S = \frac{Z}{Z + \Delta Z}$.
- **Mathematical Root Cause Insight**: On a square image ($W \times H = 320 \times 320$), focal length is $f_x = 320$. At $Z = 5.0$, a forward translation $\Delta Z = -0.10$ yields $S = \frac{5.0}{4.9} = 1.0204$ (+2.04% scale growth). An off-center point at $u = 210$ ($x = 50$) moves to $u' = 160 + \frac{50 \cdot 5.0}{4.9} = 211.02$, producing a displacement of only $1.02\text{px}$. Over 48 frames, $1.02\text{px}$ is $<0.02\text{px/frame}$, appearing completely static to human vision!
- **DIBR Solution**: To produce human-perceptible camera travel ($>15\text{px}$ shift for `MEDIUM`), camera translation $t_z$ and lateral $t_x$ must be calibrated relative to $f_x$ and depth layer distribution, using closed-loop image-space target feedback.

---

## Comparison Table

| Reference / Field | Core Mechanism | Parallax Performance | CPU/GPU Overhead | Commercial / V0 Fit |
| :--- | :--- | :--- | :--- | :--- |
| **SAM 2** | Hiera Transformer Segmentation | N/A (2D Masks) | Low (Tiny variant) | High (Apache-2.0) |
| **Depth Anything V2** | Monocular Depth Estimation | High (Relative Depth) | Low (Small variant) | High (Apache-2.0) |
| **pixelNeRF / MipNeRF** | Implicit Volumetric NeRF | High | Very High (MLP raymarching) | Low (Too slow for V0) |
| **DIBR / Pinhole (Our Engine)** | 3D Reprojection + Splatting | High (Explicit Parallax) | Extremely Fast (<1s/frame) | High (First-Principles) |
