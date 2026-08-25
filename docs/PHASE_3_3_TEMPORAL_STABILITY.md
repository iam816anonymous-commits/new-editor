# Phase 3.3 — Temporal Stability & Motion Coherence

## Executive Summary
Temporal stability evaluates frame-to-frame motion continuity and detects high-frequency flicker, temporal tearing, or erratic displacement across rendered frame sequences.

## Diagnostic Metrics
Implemented in `quality/metrics.py` (`compute_temporal_diagnostics`):
- **Frame MAD**: Mean Absolute Difference $MAD(f_n, f_{n-1}) = \frac{1}{HW}\sum |f_n - f_{n-1}|$.
- **Layer-Specific MAD**: Separate tracking for Subject, Boundary, and Background regions.
- **Loop Closure Error**: Evaluated for cyclic trajectories:
  $$\text{MAE}_{\text{loop}} = \frac{1}{HW}\sum |f_{\text{last}} - f_0|$$

## Verification Plot
Exports `temporal_diagnostics.png` visualising frame-to-frame MAD curves across the entire sequence.
