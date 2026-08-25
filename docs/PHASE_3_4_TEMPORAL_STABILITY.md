# Phase 3.4 — Temporal Stability & Subject Reprojection

## Executive Summary
Temporal instability in single-view video generation manifests as high-frequency flickering, texture deformation, or boundary warping. Phase 3.4 implements persistent background canvas tracking and confidence-weighted temporal subject reprojection.

## Persistent Background Canvas
Implemented in `rendering/disocclusion.py` (`PersistentBackgroundCanvas`):
- Maintains cross-frame background plate $RGB_{\text{bg}}$ and depth $Z_{\text{bg}}$.
- Updates canvas only when newly exposed pixels are disoccluded, preventing per-frame Telea inpainting flicker.

## Temporal Subject Reprojection
Implemented in `rendering/sequence_renderer.py`:
- Blends consecutive rendered subject keyframes $f_{t-1}$ and $f_t$:
  $$RGB_t = 0.85 \cdot RGB_t + 0.15 \cdot RGB_{t-1}$$
- High-confidence core pixels maintain 100% texture fidelity while temporal blending dampens micro-flicker along boundaries.
