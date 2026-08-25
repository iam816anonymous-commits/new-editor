# Phase 3.4 Ground Truth Audit Report

**Date:** Current System Session
**Auditor:** Jules (First-Principles AI Engineer)

---

## Symbol Audit Status

| Symbol | Location | Status | Summary |
|---|---|---|---|
| `SubjectAnchor` | `modes/mode_2_5d/scene.py`, `test_v0_pipeline.py` | IMPLEMENTED | Represents bounding box, centroid, width, height, reference depth. |
| `extract_subject_anchor` | `modes/mode_2_5d/scene.py`, `test_v0_pipeline.py` | IMPLEMENTED | Extracts anchor from subject binary mask and depth map. |
| `regularize_subject_depth` | `geometry/transforms.py`, `modes/mode_2_5d/renderer.py` | IMPLEMENTED | Applies edge-preserving Joint Bilateral Filtering / median filtering on subject depth. |
| `compute_subject_rigid_transform` | `geometry/transforms.py`, `modes/mode_2_5d/renderer.py` | IMPLEMENTED | Calculates rigid 2D SE(2) affine transform around subject anchor centroid. |
| `warp_subject_layer_rigid_subpixel` | `geometry/splatting.py`, `modes/mode_2_5d/renderer.py` | IMPLEMENTED | Float32 subpixel warp for rigid subject layer rendering. |
| `PersistentBackgroundCanvas` | `rendering/disocclusion.py` | IMPLEMENTED | Temporal persistent background canvas to avoid frame-to-frame Telea inpainting flicker. |
| `SubjectPixelIntegrityMetric` | `quality/metrics.py` | IMPLEMENTED | Data structure holding rigid residual, edge stability, texture stability metrics. |
| `compute_subject_pixel_integrity` | `quality/metrics.py` | IMPLEMENTED | Calculates rigid residual and texture stability on rendered subject keyframes. |
| `validate_subject_lock_quality_gate` | `quality/metrics.py` | IMPLEMENTED | Evaluates pass/fail thresholds for subject integrity and motion effectiveness. |
| `validate_bounding_box_contract` | `core/contracts.py` | IMPLEMENTED | Enforces `[x1, y1, x2, y2]` bounding box bounds and order contracts. |
| `RIGID_CAMERA_ATTACHED` | `core/enums.py` | IMPLEMENTED | Enum member for rigid camera-attached motion policy. |
| `PRIMARY_SUBJECT_INTERIOR` | `core/enums.py` | IMPLEMENTED | Protected layer designation for subject interior pixels. |

---

## Operational Verification
All symbols are integrated into the production pipeline (`modes/mode_2_5d/renderer.py`, `rendering/disocclusion.py`, `quality/metrics.py`) and tested via unit tests in `test_v0_pipeline.py`.
