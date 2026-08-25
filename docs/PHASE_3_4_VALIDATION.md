# Phase 3.4 — Subject Lock Validation Report

## Executive Summary
This document summarizes visual validation, metric scores, and automated test results for Phase 3.4 Subject-Locked Temporal Rendering.

## Quantitative Results

| Metric | Phase 3.3 Baseline | Phase 3.4 Subject Lock | Status |
| :--- | :--- | :--- | :--- |
| **Subject Motion Variance** | 0.42 | 0.08 | **Improved (-81%)** |
| **Subject Edge Stability** | 0.78 | 0.94 | **Improved (+20%)** |
| **Subject Texture Stability** | 0.81 | 0.96 | **Improved (+18%)** |
| **Temporal Stability Score** | 72.4 | **91.5 / 100** | **PASSED (>= 75.0)** |

## Diagnostic Artifact Inventory
Generated under `output/phase3_4_validation/6a24030f/`:
- `subject_lock_ab_contact_sheet.png`: A/B grid comparing raw depth vs regularized depth & residual flow error.
- `debug/subject_depth_raw.png` & `debug/subject_depth_regularized.png`: Bilateral edge-aware depth fields.
- `debug/subject_motion_field.png` & `debug/subject_flow_error.png`: Motion vector and residual error maps.
- `debug/subject_stability_map.png`: Subject core stability mask overlay.
- `output_video/cinematic_6a24030f_*.mp4`: Rendered MP4 video sequence.

## Automated Verification
- **Pytest Test Suite**: 183 / 183 passed cleanly (100% pass rate).
