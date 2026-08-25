# Phase 3.1 Monolith & Package Function Dependency Map

## Executive Summary
This document enumerates the function, class, data structure, and dependency relationships across the modularized First-Principles 2.5D Parallax & Inferred 3D Renderer codebase. It serves as the authoritative map verifying zero circular dependencies, strict single-responsibility boundaries, and clean mode isolation.

---

## 1. Monolith Line Count & Extraction Summary

| Metric / Dimension | Pre-Refactor Monolith | Post-Refactor State |
| --- | --- | --- |
| `v0_pipeline.py` Line Count | 4,373 lines | 217 lines |
| Package Structure | Single script + flat subpackages | 10 modular packages |
| Mode Isolation | Mixed functions in monolith | `modes/mode_2_5d/` & `modes/mode_3d/` |
| Monolithic Duplication | High | Zero duplicate renderers |

---

## 2. Component Classification & Package Mapping

### CORE CONTRACTS (`core/`)
- `core/contracts.py`: Dataclasses (`InputImage`, `DepthField`, `SubjectSelectionContract`, `SceneRepresentation`, `CameraPose`, `CameraTrajectory`, `RenderRequest`, `RenderResult`).
- `core/enums.py`: Enums (`RenderMode`, `MotionStyle`, `MotionStrength`, `QualityTier`, `AttachmentType`).
- `core/types.py`: Type aliases (`ImageArray`, `DepthArray`, `MaskArray`).
- `core/errors.py`: Custom exceptions (`PipelineError`, `ModelInferenceError`, `RenderError`).

### INFERENCE (`inference/`)
- `inference/device.py`: `get_device()` (PyTorch device discovery CUDA/CPU).
- `inference/depth.py`: `load_depth_anything_v2()`, `infer_raw_depth()`, `handle_depth_outliers_and_normalize()`, `edge_aware_depth_refinement()`, `compute_depth_confidence_map()`.
- `inference/segmentation.py`: `load_sam2()`, `segment_subject_sam2()`, `validate_subject_mask()`.
- `inference/model_manager.py`: `ModelManager` (Model caching & lifecycle management).

### GEOMETRY (`geometry/`)
- `geometry/projection.py`: `back_project_points()`, `project_3d_points()`.
- `geometry/transforms.py`: `compute_rotation_matrix()`, `transform_3d_points()`.
- `geometry/splatting.py`: `render_single_frame_forward_splatting()`.
- `geometry/zbuffer.py`: `deterministic_z_buffer_update()`.

### CAMERA (`camera/`)
- `camera/intrinsics.py`: `derive_camera_intrinsics()`.
- `camera/trajectories.py`: `generate_c1_smooth_trajectory()`, `construct_layer_motion_map()`.
- `camera/safety.py`: `plan_safe_motion_trajectory()`, `compute_safety_margins()`, `run_trajectory_magnitude_sweep()`.

### MODE A — 2.5D RENDERER (`modes/mode_2_5d/`)
- `modes/mode_2_5d/pipeline.py`: `Mode25DPipeline` orchestrator.
- `modes/mode_2_5d/scene.py`: 2.5D Scene setup and layer decomposition.
- `modes/mode_2_5d/renderer.py`: Forward splatting & Z-buffer 2.5D view synthesis.
- `modes/mode_2_5d/motion.py`: Safe closed-loop 2.5D trajectory generation.
- `modes/mode_2_5d/diagnostics.py`: Mode A crop diagnostics & contact sheets.

### MODE B — INFERRED 3D RENDERER (`modes/mode_3d/`)
- `modes/mode_3d/pipeline.py`: `Mode3DPipeline` orchestrator.
- `modes/mode_3d/reconstruction.py`: `PointCloud3D`, `MeshGeometry3D`, perspective unproject.
- `modes/mode_3d/scene_builder.py`: `Inferred3DScene` construction from depth/RGB.
- `modes/mode_3d/renderer.py`: Free-viewpoint mesh/point-cloud rasterizer & viewpoint sweep.
- `modes/mode_3d/export.py`: OBJ, PLY, and GLB file exporters.

### RENDERING PRIMITIVES (`rendering/`)
- `rendering/disocclusion.py`: `refine_and_dilate_subject_mask()`, `apply_depth_aware_edge_feathering()`, `compute_boundary_risk_map()`, `reconstruct_background_rgb()`, `complete_background_depth()`, `compute_provenance_map()`.
- `rendering/frame_renderer.py`: `verify_zero_motion_identity()`, `synthesize_micro_motion_frame()`, `run_micro_motion_sweep()`.
- `rendering/sequence_renderer.py`: `render_phase_e_representative_keyframes()`, `render_full_frame_sequence()`.
- `rendering/video_encoder.py`: `verify_ffmpeg()`, `encode_and_verify_mp4()`.

### QUALITY & DIAGNOSTICS (`quality/`)
- `quality/planner.py`: `HardwareProfile`, `QualityProfile`, `QualityDecision`, `QualityPlanner`, `SceneComplexityTier`, `SceneComplexityAnalyzer`.
- `quality/metrics.py`: `classify_motion_visibility()`, `evaluate_subject_scale_change()`, `compute_perceptual_motion_score()`, `compute_temporal_diagnostics()`, `compute_subject_rigidity_metrics()`.
- `quality/diagnostics.py`: `export_temporal_motion_profile()`, `generate_camera_vs_raster_motion_plot()`.
- `quality/hardware.py`: VRAM/CPU hardware discovery and quality matrix mapping.

### OUTPUT & ARTIFACTS (`output/`)
- `output/artifacts.py`: Directory setup (`setup_cache_directory()`, `setup_output_directories()`), artifact exporters (`save_phase_b_diagnostic_artifacts()`, `save_phase_c_diagnostic_artifacts()`, `save_phase_d_validation_artifacts()`, `save_phase_e_artifacts()`).
- `output/video.py`: MP4 encoding wrappers and keyframe PNG exporters.
- `output/manifests.py`: JSON metrics and manifest generation (`metrics.json`, `quality_report.json`).

### APPLICATION & CLI (`app/`)
- `app/cli.py`: `parse_args()` with support for all 12 CLI parameters.
- `app/application.py`: `CinematicRendererApp` workflow orchestrator.

---

## 3. Dependency Flow Graph & Boundary Rules

```
       [app/cli.py] -> [app/application.py]
                             |
             +---------------+---------------+
             |                               |
     [modes/mode_2_5d]               [modes/mode_3d]
             |                               |
             +---------------+---------------+
                             |
       +---------------------+---------------------+
       |                     |                     |
[rendering/]             [camera/]             [quality/]
       |                     |                     |
       +---------------------+---------------------+
                             |
                   [geometry/ & inference/]
                             |
                         [core/]
```

### Dependency Invariants:
1. `core/` has **zero external imports** within the codebase.
2. `geometry/` and `camera/` depend **only on `core/` and standard scientific libraries** (`numpy`, `cv2`).
3. `inference/` depends **only on `core/` and PyTorch/HuggingFace libraries**.
4. `modes/mode_2_5d` and `modes/mode_3d` are **strictly disjoint**; neither imports from the other.
5. `v0_pipeline.py` is a **thin entry point** (<300 lines) delegating to `app/`.
