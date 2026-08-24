# Phase 3.1 Codebase & Architecture Forensic Audit Report

## 1. Executive Summary
The First-Principles Cinematic 2.5D/3D Renderer has grown to over 2,400 lines in `v0_pipeline.py` and 3,700 lines in `test_v0_pipeline.py`. `v0_pipeline.py` currently mixes CLI parsing, model inference, 3D pinhole projection, forward splatting, Z-buffer compositing, closed-loop motion trajectory planning, disocclusion inpainting, FFmpeg video encoding, quality metrics calculation, and diagnostic contact sheet rendering into a single monolithic script.

This refactoring phase decomposes `v0_pipeline.py` into a modular package hierarchy without modifying any rendering mathematics or breaking the behavioral output contract.

---

## 2. Current Module Map & Responsibility Breakdown

| Module / Directory | Current Responsibilities | Lines of Code | Coupling Risk / Architectural Issues |
| --- | --- | --- | --- |
| `v0_pipeline.py` | Orchestration, CLI, model loading, depth inference, SAM 2 segmentation, pinhole math, forward splatting, Z-buffering, trajectory planning, inpainting, video encoding, perceptual quality metrics, contact sheet exports. | ~2,420 lines | **HIGH COUPLING / MONOLITHIC**. Combines all pipeline stages in one file. |
| `spatial_intelligence/` | Multi-layer depth fields, scene graph, entity consolidation, trust scoring, relationship inference, parallax region coupling, visual quality scoring. | ~2,800 lines | **PROTECTED MATURE MODULE**. Clean internal boundaries; reuse as-is. |
| `subject_selection/` | Multi-signal candidate feature extraction, scoring, compound candidate grouping, edge-constrained mask refinement, confidence gate. | ~1,600 lines | **PROTECTED MATURE MODULE**. Clean internal boundaries; reuse as-is. |
| `scene_3d/` | Inferred 3D scene representation, camera models, point cloud, mesh geometry, complexity analysis, quality planner. | ~1,200 lines | **PROTECTED MATURE MODULE**. Clean internal boundaries; reuse as-is. |
| `render_backend/` | Explicit 3D mesh triangulation and OBJ/PLY/GLB export. | ~250 lines | **PROTECTED MATURE MODULE**. Clean internal boundaries; reuse as-is. |

---

## 3. `v0_pipeline.py` Responsibility Breakdown & Largest Functions

1. `render_single_frame_forward_splatting` (~115 lines): Forward subpixel splatting and Z-buffer update logic.
2. `plan_safe_motion_trajectory` (~85 lines): Closed-loop safety trajectory planner.
3. `compute_perceptual_motion_score` (~110 lines): Optical flow measurement and quality gate validation.
4. `generate_c1_smooth_trajectory` (~70 lines): Camera translation and rotation curve generation.
5. `main` (~220 lines): CLI execution and pipeline orchestration.
6. Diagnostic contact sheet generators (~400 lines total across 8 functions).

---

## 4. Proposed Target Modular Architecture

```
project_root/
│
├── app/                  # Application orchestrator & CLI entry points
│   ├── cli.py
│   └── application.py
│
├── core/                 # Typed dataclasses, interfaces, error definitions
│   ├── contracts.py
│   ├── enums.py
│   └── config.py
│
├── inference/            # Model loading, device discovery, depth/SAM2 inference
│   ├── device.py
│   ├── depth.py
│   ├── segmentation.py
│   └── model_manager.py
│
├── camera/               # Pinhole intrinsics, trajectory curves, closed-loop safety
│   ├── intrinsics.py
│   ├── trajectories.py
│   └── safety.py
│
├── geometry/             # Pure math: 3D back-projection, SE(3) transforms, splatting, Z-buffer
│   ├── projection.py
│   ├── transforms.py
│   ├── splatting.py
│   └── zbuffer.py
│
├── modes/                # Explicit mode engines and router
│   ├── base.py
│   ├── mode_2_5d.py
│   ├── mode_3d.py
│   └── router.py
│
├── rendering/            # Frame synthesis, sequence rendering, inpainting, video encoding
│   ├── frame_renderer.py
│   ├── sequence_renderer.py
│   ├── disocclusion.py
│   └── video_encoder.py
│
├── quality/              # Hardware quality planning, metrics, diagnostic gates
│   ├── planner.py
│   ├── metrics.py
│   └── diagnostics.py
│
├── output/               # Artifact management, contact sheets, JSON manifests
│   └── artifacts.py
│
├── spatial_intelligence/ # Protected mature package
├── subject_selection/    # Protected mature package
├── scene_3d/             # Protected mature package
├── render_backend/       # Protected mature package
│
└── v0_pipeline.py        # Thin Orchestrator / Entry Point (< 500 lines)
```

---

## 5. Protected Modules (Do NOT Modify Unnecessarily)
- `spatial_intelligence/`
- `subject_selection/`
- `scene_3d/`
- `render_backend/`

---

## 6. Migration Safeguards & Parity Contract
- **Test Invariant:** All 181 automated tests in `test_v0_pipeline.py` must pass continuously throughout extraction.
- **Output Parity:** Input image $\to$ SHA-256 hash $\to$ identical directory structure, identical metrics JSON schemas, identical video output specifications.
