# Phase 2.9 — Forensic Codebase Audit & Mode Separation Verification Report

## 1. Executive Summary
This document provides a line-by-line forensic audit of the codebase (`v0_pipeline.py`, `scene_3d/`, `render_backend/`, `spatial_intelligence/`, `subject_selection/`), answering the 8 core inspection questions and verifying strict separation between **MODE A (2.5D Cinematic)** and **MODE B (Inferred 3D Scene)**.

---

## 2. Answers to 8 Core Forensic Inspection Questions

### Question 1: Which renderer actually produces the final MP4?
* **Answer:** `render_full_frame_sequence()` in `v0_pipeline.py` calls `render_single_frame_forward_splatting()`, which synthesizes individual keyframe PNGs before FFmpeg encodes `output.mp4`.

### Question 2: Is `scene_3d/` genuinely used for final rendering or only export/prototype paths?
* **Answer:** `scene_3d/` provides the explicit 3D scene orchestrator (`Inferred3DScene`), point cloud backprojection (`PointCloud3D`), depth mesh triangulation (`MeshGeometry3D`), 3DGS parameter prediction scaffold (`GaussianScene`), novel view point cloud rasterizer (`Inferred3DRenderer`), and 3D package export (`export_3d_package`). In Mode B, `scene_3d` directly generates 3D geometry and renders novel views.

### Question 3: Do Mode A (2.5D) and Mode B (Inferred 3D) produce materially different pixels?
* **Answer:** **YES.**
  - **Mode A (2.5D Cinematic):** Uses continuous local depth fields, depth-quantile layer motion maps, subpixel forward splatting with Z-ownership resets, and Telea background disocclusion inpainting.
  - **Mode B (Inferred 3D Scene):** Backprojects input pixels into explicit 3D point clouds and triangulated meshes, rendering novel views directly from 3D camera frustums.

### Question 4: Is the current implementation mixing representations?
* **Answer:** In Phase 2.8, the term "HYBRID" referred to CPU control logic combined with CUDA neural model inference. In Phase 2.9, "HYBRID" is strictly eliminated as a final rendering mode. Preprocessing models (Depth Anything V2, SAM 2) are shared, but the spatial rendering engine is strictly single-mode: either Mode A or Mode B.

### Question 5: Can the backend router accidentally select a hybrid render path?
* **Answer:** **NO.** The router (`QualityPlanner` / `SceneComplexityAnalyzer`) returns a single discrete decision: `MODE_A_2_5D` or `MODE_B_INFERRED_3D`.

### Question 6: Are quality/resolution decisions actually enforced?
* **Answer:** **YES.** `QualityPlanner.plan()` enforces hard CPU ceilings ($\le 720p$) and VRAM safety limits, exporting decisions to `quality_decision.json`.

### Question 7: Do CPU and CUDA execute genuinely different hardware paths?
* **Answer:** **YES.** CPU execution runs on PyTorch/NumPy CPU operators, while CUDA execution offloads model inference and tensor splatting to PyTorch CUDA devices.

### Question 8: Do reported metrics correspond to actual final video pixels?
* **Answer:** **YES.** Optical flow ($d_{\text{observed}}$), surface residual flow error ($e_{\text{residual}}$), temporal flicker, and subject scale growth are measured directly from actual rasterized output keyframe arrays ($F_0 \dots F_{\text{end}}$).
