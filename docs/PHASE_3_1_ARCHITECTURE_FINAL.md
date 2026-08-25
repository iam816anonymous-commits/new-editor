# Phase 3.1 Architecture Final Documentation

## Architectural Overview
The First-Principles Cinematic 2.5D/3D Parallax Renderer is organized into clean, single-responsibility Python packages following a strict top-down dependency flow:

```
app (CLI & Pipeline Entry Point)
  ↓
modes (Mode Router, Mode A 2.5D, Mode B 3D)
  ↓
rendering / scene_3d / render_backend
  ↓
camera / geometry / inference
  ↓
core (Contracts, Enums, Types, Errors)
```

## Package Responsibilities

### 1. `app/`
Contains the CLI parser (`cli.py`) and top-level pipeline application runner (`application.py`).

### 2. `core/`
Holds immutable typed contracts (`contracts.py`), enumerations (`enums.py`), type aliases (`types.py`), and custom exception hierarchies (`errors.py`).

### 3. `inference/`
Manages GPU/CPU device detection (`device.py`), model loading for Depth Anything V2 & SAM 2 (`model_manager.py`), depth normalization & bilateral edge refinement (`depth.py`), and SAM 2 candidate mask generation & scoring (`segmentation.py`).

### 4. `geometry/`
Provides pinhole camera projection math (`projection.py`), 3D SE(3) rotation matrices and rigid body transforms (`transforms.py`), Z-buffered forward subpixel splatting (`splatting.py`), and deterministic depth buffer compositing (`zbuffer.py`).

### 5. `camera/`
Computes camera intrinsics (`intrinsics.py`), generates C1 smooth trajectory poses (`trajectories.py`), and enforces closed-loop camera motion safety limits (`safety.py`).

### 6. `rendering/`
Implements disocclusion inpainting for clean background plates (`disocclusion.py`), zero-motion identity and single-frame synthesis (`frame_renderer.py`), temporal sequence rendering (`sequence_renderer.py`), and FFmpeg MP4 encoding (`video_encoder.py`).

### 7. `quality/`
Measures subject rigidity, perceptual motion scores, and temporal stability (`metrics.py`), generates diagnostic contact sheets and keyframe sheets (`diagnostics.py`), plots trajectory and displacement curves (`plots.py`), plans hardware quality profiles (`planner.py`), and profiles hardware capabilities (`hardware.py`).

### 8. `output/`
Calculates file hashes, loads input images, configures output directories, and saves diagnostic artifact packages (`artifacts.py`), video file verification (`video.py`), and render manifests (`manifests.py`).

### 9. `modes/`
Houses mode selection routing (`router.py`), Mode A 2.5D parallax pipeline (`mode_2_5d/`), and Mode B inferred 3D scene pipeline (`mode_3d/`).
