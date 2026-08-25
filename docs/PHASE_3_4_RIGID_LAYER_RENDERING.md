# Phase 3.4 Rigid Layer Rendering Specification

## Overview
Subpixel rigid subject layer rendering separates primary subject transformation from environmental depth splatting.

## Implementation Details
1. **Float32 Affine Matrix Formulation**: Calculates 2D rigid transform around subject anchor centroid.
2. **Subpixel Interpolation**: Uses `warp_subject_layer_rigid_subpixel` (`cv2.warpAffine` with `INTER_LINEAR`) on subject RGBA.
3. **Internal Texture Stability**: Eliminates independent per-pixel depth displacement inside the subject layer.
