# Phase 3.3 — Camera Model, Convergence Depth & Overscan Management

## Executive Summary
Perspective view synthesis requires calibrating camera focal lengths, adaptive zero-parallax convergence planes, and overscan crop boundaries to ensure distortion-free parallax and seamless frame borders.

## Adaptive Convergence Depth
Implemented in `camera/intrinsics.py` (`compute_adaptive_convergence_depth`):
- Measures subject depth median $Z_{\text{conv}} = \text{median}(Z_{\text{subject}})$.
- Positions the zero-parallax plane at the primary subject to keep subject scale stable during camera movement while surrounding environmental layers exhibit depth-proportional motion.

## Overscan Crop Management
Implemented in `camera/safety.py` (`compute_overscan_crop`):
- Evaluates peak camera trajectory disparity $d_{\text{max}}$.
- Computes overscan border crop coordinates:
  $$\text{Pad} = \lceil d_{\text{max}} \cdot 1.2 \rceil$$
  $$\text{Crop} = (y_1, x_1, y_2, x_2)$$
- Crop margins remove unrendered canvas borders, maintaining 100% valid pixel coverage across all frames.
