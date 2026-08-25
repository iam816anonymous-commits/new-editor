# Phase 3.4 — Camera vs. Subject Motion Separation

## Executive Summary
Achieving Immersity-style cinematic parallax requires decoupling background camera motion from primary subject motion. While environmental layers (foreground, midground, background) exhibit depth-proportional camera travel, the primary subject must undergo restrained, global SE(3) rigid transformation to remain internally coherent.

## Mathematical Formulation
In `geometry/transforms.py` (`compute_subject_rigid_transform`):

For camera translation $t_{\text{cam}}$ and rotation $R_{\text{cam}}$:
1. **Translation Restraint**: $t_{\text{subj}} = 0.35 \cdot t_{\text{cam}}$, scaling down lateral drift and depth expansion.
2. **Rotation Restraint**: $R_{\text{subj}} = I + 0.50 \cdot (R_{\text{cam}} - I)$, dampening angular tilt.

## Hierarchical Rendering
In `modes/mode_2_5d/renderer.py`:
- Background layer: Full camera transformation $P' = R_{\text{cam}} \cdot P + t_{\text{cam}}$.
- Subject layer: Regularized depth + restrained rigid transform $P' = R_{\text{subj}} \cdot P + t_{\text{subj}}$.

This separation ensures the viewer perceives a camera physically moving around a stable 3D subject rather than an image being warped.
