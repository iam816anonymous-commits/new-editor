# Phase 3 State-of-the-Art (SOTA) Monocular 2.5D/3D Research & Decision Table

## Executive Summary
This document conducts technical research into state-of-the-art monocular depth estimation, 3D scene reconstruction, novel-view synthesis, and disocclusion completion methods for the First-Principles Cinematic 2.5D/3D Renderer.
It establishes an explicit candidate evaluation matrix and delivers official engineering decisions (`ADOPT`, `REJECT`, `OPTIONAL`, `FUTURE`).

---

## 1. Monocular Depth Estimation Methods

### 1.1 Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`)
- **Category:** Monocular Depth / Disparity Estimation
- **Architecture:** Vision Transformer (ViT-Small) backbone with relative depth regression.
- **CPU Feasibility:** HIGH (~1.1s load, ~0.8s inference on 4-core CPU).
- **VRAM Requirements:** ~1.2 GB VRAM.
- **Licensing:** Apache-2.0 (Commercial Use Permitted).
- **Single-Image Suitability:** Ideal.
- **Verdict:** **ADOPT** (Current Production Monocular Depth Engine).

### 1.2 Video Depth Anything (`depth-anything/Video-Depth-Anything-Small`)
- **Category:** Video Monocular Depth Estimation
- **Single-Image Suitability:** Low (requires temporal sequence context).
- **Verdict:** **REJECT** (V0/V1 input is a single static 2D image).

### 1.3 Marigold / Diffusion-Based Depth
- **Category:** Latent Diffusion Depth Estimation
- **Inference Latency:** High (~15-30s per frame on CPU / ~2s on high-end GPU).
- **CPU Feasibility:** LOW.
- **Verdict:** **FUTURE** (Investigate for high-end offline GPU rendering tier only).

### 1.4 ZoeDepth
- **Category:** Metric Relative Depth
- **Licensing:** MIT License.
- **Inference Latency:** Moderate (~2.5s on CPU).
- **Verdict:** **OPTIONAL** (Viable drop-in alternative for metric depth scale calibration).

---

## 2. 3D Scene Reconstruction & Novel View Synthesis

### 2.1 Monocular Point Cloud & Depth Mesh Triangulation (`scene_3d/`)
- **Category:** Classical DIBR & Explicit 3D Geometry Reconstruction
- **Representation:** Vertices [X, Y, Z], Colors [R, G, B], UVs, Quad Triangulation with Discontinuity Breaking.
- **CPU Feasibility:** HIGH (~0.2s CPU execution).
- **Licensing:** In-house First-Principles (MIT / Open).
- **Single-Image Suitability:** Excellent.
- **Verdict:** **ADOPT** (Current Mode B Explicit 3D Reconstruction Engine).

### 2.2 DUSt3R / MASt3R
- **Category:** Unconstrained Multi-View 3D Point Cloud Regression
- **Single-Image Suitability:** Low to Moderate (designed for uncalibrated image pairs).
- **VRAM Requirements:** ~8-12 GB VRAM.
- **Verdict:** **REJECT** for V0/V1 local CPU/GPU execution due to high VRAM footprint and multi-view assumption.

### 2.3 3D Gaussian Splatting (3DGS) / InstantSplat
- **Category:** Radiance Field Rasterization
- **Single-Image Suitability:** Requires unobserved surface hallucination.
- **CPU Feasibility:** LOW (CUDA rasterization kernels required).
- **Verdict:** **FUTURE** (Scaffold parameter prediction retained in `scene_3d/gaussian_scene.py` for GPU-enabled Phase 4).

### 2.4 Apple SHARP / VGGT
- **Category:** Feedforward Single-Image 3D Geometry
- **Licensing:** Proprietary / Non-commercial research restrictions.
- **Verdict:** **REJECT** (Violates commercial licensing non-negotiables).

---

## 3. Disocclusion & Inpainting Methods

### 3.1 Navier-Stokes / Telea Boundary Propagation (`cv2.inpaint`)
- **Category:** Classical Image Inpainting
- **CPU Feasibility:** EXTREMELY HIGH (<0.05s CPU runtime).
- **Licensing:** BSD-3-Clause (OpenCV).
- **Single-Image Suitability:** Excellent for background plate completion behind subject masks.
- **Verdict:** **ADOPT** (Current Production Disocclusion Inpainting Engine).

### 3.2 Generative Diffusion Inpainting (LaMa / Stable Diffusion Inpainting)
- **Category:** Deep Generative Completion
- **CPU Feasibility:** LOW (>10s CPU latency).
- **Verdict:** **FUTURE** (Investigate for high-VRAM GPU modes).

---

## 4. Official SOTA Decision Table

| Method Name | Category | CPU Feasibility | Licensing | Decision | Primary Reason |
| --- | --- | --- | --- | --- | --- |
| **Depth Anything V2 Small** | Depth | HIGH (~0.8s) | Apache-2.0 | **ADOPT** | Fast, high-edge monocular depth precision, lightweight footprint. |
| **SAM 2 Hiera-Tiny** | Segmentation | HIGH (~0.5s) | Apache-2.0 | **ADOPT** | Precise promptable subject segmentation with boundary fidelity. |
| **Pinhole Backprojection + Subpixel Splatting** | 2.5D Render | HIGH (~0.1s) | Open / First-Principles | **ADOPT** | Zero-motion exact reconstruction, deterministic Z-buffer compositing. |
| **Depth Mesh Triangulation** | 3D Geometry | HIGH (~0.2s) | Open / First-Principles | **ADOPT** | Explicit 3D mesh reconstruction with OBJ/PLY/GLB export. |
| **Navier-Stokes / Telea Inpainting** | Inpainting | HIGH (<0.05s) | BSD-3-Clause | **ADOPT** | Deterministic, fast disocclusion completion without generative hallucination. |
| **ZoeDepth** | Depth | MEDIUM (~2.5s) | MIT | **OPTIONAL** | Metric scale calibration alternative. |
| **Marigold** | Depth | LOW (>15s CPU) | Apache-2.0 | **FUTURE** | High GPU latency; consider for offline ultra-quality preset. |
| **3D Gaussian Splatting** | Radiance Field | LOW (CUDA only) | MIT | **FUTURE** | Requires GPU CUDA rasterizer; scaffolded for Phase 4. |
| **DUSt3R / MASt3R** | 3D Recon | LOW (>8GB VRAM) | CC-BY-NC 4.0 | **REJECT** | Non-commercial license and multi-view input requirement. |
| **Apple SHARP / VGGT** | 3D Recon | LOW | Non-commercial | **REJECT** | Non-commercial licensing restrictions. |
| **Video Depth Anything** | Depth | LOW | Apache-2.0 | **REJECT** | Requires multi-frame video input stream. |

---

## 5. Dependency Non-Explosion Rule
The selected pipeline dependencies remain strictly bounded:
- `torch` & `transformers` (PyTorch compute + Depth Anything V2)
- `sam2` & `huggingface_hub` (SAM 2 segmentation)
- `opencv-python` & `numpy` (Spatial operations, classical inpainting, optical flow)
- `pillow` (Image I/O)

No external heavyweight multi-gigabyte frameworks or non-commercial binaries have been introduced.
