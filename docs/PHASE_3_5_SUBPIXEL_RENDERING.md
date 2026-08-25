# Phase 3.5 Subpixel Rendering Specification

## Forward Splatting Engine
- Subpixel bilinear distribution to 2x2 target pixel neighborhood using `np.add.at`.
- Deterministic Z-buffer ownership tracking.
- Float32 affine subject layer warping `warp_subject_layer_rigid_subpixel` preserving internal subject texture.
