# Phase 3.1 Output Pipeline Final Repair Report

## 1. Measured Evidence & Physical Verification

- **v0_pipeline.py Physical Line Count:** 49 lines
- **Frames Generated:** YES (48 frames generated in `cinematic/frames/frame_0000.png` .. `frame_0047.png`)
- **Frame Count:** 48
- **FFmpeg Detected:** YES (`ffmpeg version 7.0.2-static`)
- **FFmpeg Return Code:** 0
- **MP4 Generated:** YES
- **MP4 Output Path:** `output/<short_hash>/cinematic/output.mp4`
- **MP4 Byte Size:** 121,942 bytes
- **FPS:** 24.0
- **Resolution:** 320x320
- **Video Verification Result:** OpenCV VideoCapture verified frame count = 48, FPS = 24.0, duration = 2.0s
- **Manifests Generated:** `metrics.json`, `motion_report.json`, `validation_summary.json`
- **Mode A (2.5D) Result:** Verified
- **Mode B (3D) Result:** Verified
- **Auto Mode Result:** Verified
- **Pytest Result:** 181 / 181 passed (100% pass rate)

---

## 2. Directory Structure Verification

```
output/<short_hash>/
├── 01_original.png
├── 02_depth.png
├── 03_candidate_contact_sheet.png
├── 04_candidate_scores.png
├── background_depth.png
├── background_plate.png
├── boundary_risk_map.png
├── camera_path.json
├── camera_path.png
├── camera_vs_raster_motion.png
├── candidate_selection.json
├── confidence_map.png
├── depth.png
├── final_contact_sheet.png
├── layer_displacement_curves.png
├── metrics.json
├── motion_amplitude_comparison.png
├── motion_report.json
├── original.png
├── phase_1_7_visual_validation_contact_sheet.png
├── provenance_map.png
├── spatial_diagnostics.json
├── spatial_relationships.json
├── spatial_scene.json
├── subject_mask.png
├── temporal_diagnostics.png
├── validation_summary.json
├── subtle/
├── strong/
├── cinematic/
│   ├── output.mp4
│   ├── visual_review.png
│   ├── visual_review_diagnostics.png
│   └── frames/
│       ├── frame_0000.png
│       └── ...
├── debug/
│   ├── depth_distribution.json
│   ├── frame_diff_f00_f99.png
│   ├── frame_difference_report.json
│   ├── frame_overlay_f00_f99.png
│   ├── motion_heatmap.png
│   ├── pixel_trajectory.png
│   ├── projected_vs_raster_trajectory.png
│   ├── raster_trace.json
│   └── temporal_motion_profile.json
└── spatial_analysis/
    ├── 00_primary_spatial_overlay.png
    ├── 01_depth_raw.png
    ├── 02_depth_edges.png
    ├── 15_splat_coverage.png
    ├── 16_zbuffer_ownership.png
    ├── attachment_graph.json
    ├── disocclusion_forecast.json
    ├── motion_coupling.json
    ├── motion_eligibility.json
    ├── parallax_regions.json
    └── scene_analysis.json
```
