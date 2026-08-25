# Phase 3.1 Output Directory Forensics Report

## 1. Directory Lifecycle & Hierarchy Comparison

```
ORIGINAL MONOLITH (Pre-Refactor)
CLI
 ↓
v0_pipeline.py:setup_output_directories
 ↓
hash_dir: output/<short_hash>/
├── subtle/
├── cinematic/
├── strong/
├── debug/
└── spatial_analysis/
 ↓
v0_pipeline.py:render_full_frame_sequence
 ↓
frames_dir: output/<short_hash>/<strength>/frames/
 ↓
v0_pipeline.py:encode_and_verify_mp4
 ↓
output_mp4: output/<short_hash>/<strength>/output.mp4

---

MODULAR DECOUPLED (Post-Repair)
CLI
 ↓
app/cli.py -> app/application.py:run_pipeline
 ↓
output/artifacts.py:setup_output_directories
 ↓
hash_dir: output/<short_hash>/
├── subtle/
├── cinematic/
├── strong/
├── debug/
└── spatial_analysis/
 ↓
modes/mode_2_5d/pipeline.py / modes/mode_3d/pipeline.py
 ↓
rendering/sequence_renderer.py:render_full_frame_sequence
 ↓
frames_dir: output/<short_hash>/<strength>/frames/
 ↓
rendering/video_encoder.py:encode_and_verify_mp4
 ↓
output_mp4: output/<short_hash>/<strength>/output.mp4
```

## 2. Directory Lifecycle Operation Matrix

| Operation | Pre-Refactor Location | Post-Refactor Location | Execution Status | Directory Produced |
|-----------|------------------------|------------------------|------------------|-------------------|
| Root Hash Dir | `v0_pipeline.py:setup_output_directories` | `output/artifacts.py:setup_output_directories` | Called | `output/<hash>/` |
| Motion-Strength Dirs | `v0_pipeline.py:setup_output_directories` | `output/artifacts.py:setup_output_directories` | Called | `subtle/`, `cinematic/`, `strong/` |
| Debug Directory | `v0_pipeline.py` debug trace exports | `quality/plots.py`, `quality/diagnostics.py` | Called | `output/<hash>/debug/` |
| Spatial Analysis Dir | `spatial_engine.py` | `spatial_intelligence/spatial_engine.py` | Called | `output/<hash>/spatial_analysis/` |
| Frame Sequence Dir | `v0_pipeline.py:render_full_frame_sequence` | `rendering/sequence_renderer.py` | Called | `output/<hash>/<strength>/frames/` |
| Final MP4 Video File | `v0_pipeline.py:encode_and_verify_mp4` | `rendering/video_encoder.py` | Called | `output/<hash>/<strength>/output.mp4` |

---

## 3. Physical Directory Audit Results

- **Output Root:** `output/debug_post_refactor/6a24030f/`
- **Subdirectories Created:** `subtle/`, `cinematic/`, `strong/`, `debug/`, `spatial_analysis/`, `cinematic/frames/`
- **Frames Generated:** 48 frames (`frame_0000.png` through `frame_0047.png`)
- **Video Output:** `cinematic/output.mp4` (121,942 bytes, verified via OpenCV VideoCapture)
