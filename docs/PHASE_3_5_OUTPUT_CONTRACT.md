# Phase 3.5 Output Contract Specification

## Artifact Directory Hierarchy
Output artifacts strictly placed under `output/<short_hash>/`:
- `original.png`: Original reference copy
- `depth.png`, `subject_mask.png`, `confidence_map.png`: Diagnostic artifacts
- `background_plate.png`, `background_depth.png`, `provenance_map.png`: Disocclusion artifacts
- `metrics.json`, `motion_report.json`, `validation_summary.json`: Diagnostic reports
- `output_video/cinematic_<hash>_<timestamp>.mp4`: Unique timestamped H.264 video
