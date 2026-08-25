# Phase 3.1 Output Pipeline Forensic Report

## 1. Execution Flow & Trace Analysis

The complete execution path from entry point to final output artifacts:

```
v0_pipeline.py
  ↓
app/cli.py (parse_args / main)
  ↓
app/application.py (run_pipeline)
  ↓
modes/router.py (select_render_mode)
  ↓
modes/mode_2_5d/pipeline.py (Mode25DPipeline) OR modes/mode_3d/pipeline.py (Mode3DPipeline)
  ↓
rendering/sequence_renderer.py (render_full_frame_sequence)
  ↓
rendering/video_encoder.py (encode_and_verify_mp4)
  ↓
output/artifacts.py (save_phase_b_diagnostic_artifacts, save_phase_c_diagnostic_artifacts, save_phase_e_artifacts)
  ↓
Final Output Directory (output/<short_hash>/<strength>/output.mp4)
```

## 2. Call Graph & Modular Behavior Matrix

| Original Monolith Step | Pre-Refactor Location | Post-Refactor Location | Execution Status | Equivalent? |
|------------------------|-----------------------|------------------------|------------------|-------------|
| CLI Argument Parsing | `v0_pipeline.py:parse_args` | `app/cli.py:parse_args` | Called | YES |
| Image Load & SHA256 | `v0_pipeline.py:validate_and_load_image` | `output/artifacts.py:validate_and_load_image` | Called | YES |
| Output Dir Setup | `v0_pipeline.py:setup_output_directories` | `output/artifacts.py:setup_output_directories` | Called | YES |
| Device Selection | `v0_pipeline.py:get_device` | `inference/device.py:get_device` | Called | YES |
| Depth Inference | `v0_pipeline.py:infer_raw_depth` | `inference/depth.py:infer_raw_depth` | Called | YES |
| SAM2 Segmentation | `v0_pipeline.py:segment_subject_sam2` | `inference/segmentation.py:segment_subject_sam2` | Called | YES |
| Disocclusion Inpainting | `v0_pipeline.py:reconstruct_background_rgb` | `rendering/disocclusion.py:reconstruct_background_rgb` | Called | YES |
| Spatial Intelligence | `v0_pipeline.py:analyze_spatial_scene` | `spatial_intelligence/spatial_engine.py` | Called | YES |
| Trajectory Safety Plan | `v0_pipeline.py:plan_safe_motion_trajectory` | `camera/safety.py:plan_safe_motion_trajectory` | Called | YES |
| Mode Selection & Dispatch | `v0_pipeline.py` inline | `modes/router.py` + `modes/mode_2_5d/pipeline.py` / `modes/mode_3d/pipeline.py` | Called | YES |
| Frame Sequence Rendering | `v0_pipeline.py:render_full_frame_sequence` | `rendering/sequence_renderer.py` | Called | YES |
| MP4 Video Encoding | `v0_pipeline.py:encode_and_verify_mp4` | `rendering/video_encoder.py` | Called | YES |
| Artifact Persistence | `v0_pipeline.py` inline | `output/artifacts.py` & `quality/diagnostics.py` | Called | YES |

---

## 3. Physical Output Verification

- **Frame Generation:** 48 PNG frames generated under `output/<short_hash>/cinematic/frames/frame_0000.png` .. `frame_0047.png`.
- **FFmpeg MP4 Video:** `output/<short_hash>/cinematic/output.mp4` generated and verified via OpenCV VideoCapture (121,942 bytes).
- **Manifests:** `metrics.json`, `motion_report.json`, `validation_summary.json` fully populated.
- **Mode A (2.5D):** Verified.
- **Mode B (3D):** Verified.
- **Auto Mode:** Verified.
