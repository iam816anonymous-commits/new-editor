# Phase 3.3 — Pre-Animation Hidden Region Inpainting & Edge Extrusion

## Executive Summary
To prevent spatial tearing, rubber-sheeting, or black disocclusion holes during camera motion, Phase 3.3 implements pre-animation hidden region inpainting and occlusion edge padding extrapolation in `rendering/disocclusion.py`.

## Core Implementation
- **`inpaint_hidden_regions(image, mask, method="telea", radius=5)`**: Pre-inpaints missing or masked regions in both RGB images and single-channel depth maps using Fast Marching Telea or Navier-Stokes boundary propagation.
- **`extrapolate_edge_padding(image, edge_mask, pad_size=15)`**: Extrapolates background texture and depth into edge padding zones surrounding high-gradient occlusion boundaries.

## Provenance Tracking
`compute_provenance_map` tracks pixel authenticity throughout inpainting:
- $1.0$: Observed reference pixel
- $0.0$: Inpainted / reconstructed pixel

Inpainted regions are assigned lower confidence weights during Z-buffered forward splatting to ensure observed geometric features take precedence over interpolated boundaries.
