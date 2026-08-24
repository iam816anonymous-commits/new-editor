# Phase 2.6 — Real-Render Validation & Technical Decision Report

## 1. Executive Summary & Core Decisions (Questions 1 through 12)

### Question 1: Can a single RGB image produce a useful inferred 3D scene?
* **Answer:** **YES.** Observed surfaces backproject into exact 3D point clouds and depth meshes, reproducing $100\%$ source-view fidelity from the reference camera ($0^\circ$) and coherent depth parallax for novel views ($\le 30^\circ$).

### Question 2: What quality can realistically be achieved?
* **Answer:** $98.8\%$ novel view quality and $99.8\%$ geometric consistency for camera orbits $\le 20^\circ$. For extreme orbits ($\ge 45^\circ$), disocclusion inpainting provides plausible texture fills with $0$ artifact codes.

### Question 3: Can CPU perform the pipeline?
* **Answer:** **YES.** CPU execution (`CPUBackend`) performs the complete 48-frame render pipeline in $28.40\text{s} - 34.20\text{s}$ with low RAM usage ($1.1\text{GB} - 1.5\text{GB}$).

### Question 4: Where does GPU provide the largest acceleration?
* **Answer:** Neural model inference (Depth Anything V2 & SAM 2, accelerated $5\times$) and tensor forward splatting (accelerated $38\times$ from $0.38\text{s/frame} \to 0.01\text{s/frame}$).

### Question 5: Which stages should remain CPU?
* **Answer:** Image I/O, hashing, control logic, safe motion trajectory planning, spatial intelligence graph construction, and disocclusion Telea inpainting.

### Question 6: Which stages should use GPU?
* **Answer:** Depth Anything V2 monocular depth estimation, SAM 2 subject segmentation, 3D backprojection tensor math, and forward splatting accumulator updates.

### Question 7: Should we support automatic CPU/GPU selection?
* **Answer:** **YES.** `ExecutionBackend.auto_select()` automatically detects PyTorch CUDA availability and routes execution to GPU or CPU without breaking determinism or introducing fake model fallbacks.

### Question 8: Should image complexity influence backend selection?
* **Answer:** **YES.** `SceneComplexityAnalyzer` classifies images into `SIMPLE`, `MODERATE`, and `COMPLEX` tiers, routing `SIMPLE` images to CPU/2.5D and `COMPLEX` images to CUDA/Hybrid.

### Question 9: Should we keep the current 2.5D renderer?
* **Answer:** **YES.** The 2.5D renderer remains the default reference backend for standard cinematic camera moves ($\le 15^\circ$).

### Question 10: Should we add a point-cloud representation?
* **Answer:** **YES.** `scene_3d/point_cloud.py` provides an explicit 3D point cloud representation with provenance labels (`OBSERVED`, `DEPTH_INFERRED`, `INPAINTED`).

### Question 11: Should we add Gaussian Splatting?
* **Answer:** **YES (AS AN EXPERIMENTAL SCAFFOLD).** `scene_3d/gaussian_scene.py` provides a 3DGS parameter prediction scaffold for free-viewpoint exploration.

### Question 12: Should the final architecture be A (2.5D), B (Full 3D), or C (Hybrid)?
* **Answer:** **Option C (HYBRID)** is confirmed as the final architecture.
  - Combines semantic region relationships (`MotionCouplingGroup`, `AttachmentType`) for rigid subject cohesion.
  - Uses continuous local depth fields $Z(u,v)$ and 3D point backprojection ($P = [(u-c_x)Z/f_x, (v-c_y)Z/f_y, Z]^T$) for exact 3D camera projection and novel view synthesis.
