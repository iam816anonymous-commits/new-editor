# Phase 3.3 — Quality Gate & Validation Suite

## Executive Summary
Phase 3.3 establishes automated quality contracts enforcing subject rigidity, temporal smoothness, disocclusion hole control, and motion visibility verification across all Mode A renders.

## Quality Criteria
Implemented in `quality/metrics.py` (`validate_neural_layered_render_quality`):
1. **Motion Visibility Class**: Motion must be `SUBTLE`, `VISIBLE`, or `CINEMATIC` (rejects `WEAK` and `UNSAFE`).
2. **Subject Rigidity**: Local coherence / structural stability $> 0.70$.
3. **Temporal Instability**: Overall temporal MAD $< 25.0$.
4. **Disocclusion Holes**: Inpainted/reconstructed area $< 12.0\%$.

## Verification Results
- **Unit Test Suite**: 182 / 182 passed cleanly (`pytest`).
- **Real Execution**: Successfully rendered `test_assets/test.jpeg` (320x320) with valid MP4 output and contact sheet `output/phase3_3_validation/6a24030f/neural_layered_contact_sheet.png`.
