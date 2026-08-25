# Phase 3.4 — Output Contract & File Lifecycle

## Executive Summary
Phase 3.4 enforces output contracts for artifact generation, video encoding, and diagnostic directory structures during Subject-Locked 2.5D view synthesis.

## Folder Hierarchy & Contracts
When running `--render-video`, the pipeline generates and validates:

```
output/<short_hash>/
├── original.png
├── depth.png
├── subject_mask.png
├── confidence_map.png
├── background_plate.png
├── background_depth.png
├── provenance_map.png
├── boundary_risk_map.png
├── metrics.json
├── subject_lock_ab_contact_sheet.png
├── debug/
│   ├── subject_depth_raw.png
│   ├── subject_depth_regularized.png
│   ├── subject_motion_field.png
│   ├── subject_flow_error.png
│   ├── subject_stability_map.png
│   ├── disocclusion_mask.png
│   └── temporal_difference.png
└── output_video/
    └── cinematic_<short_hash>_<timestamp>.mp4
```

## Contract Enforcement
`validate_output_contract` in `output/artifacts.py` asserts that all required PNG diagnostic maps and MP4 video renders exist and are non-empty (>0 bytes), raising `OutputContractError` if any artifact is missing.
