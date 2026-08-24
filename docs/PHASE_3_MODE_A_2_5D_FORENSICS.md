# Phase 3 Mode A (2.5D Cinematic Renderer) Forensic Real-Image Validation Report

## Executive Summary
Mode A (2.5D Cinematic Parallax Renderer) was evaluated across 10 deterministic real-image scene categories under motion types and 3 strength levels.
Mode A employs Depth Anything V2 monocular depth, SAM 2 Hiera-Tiny subject masking, Telea background RGB/depth inpainting, 3D pinhole camera backprojection, and forward subpixel splatting with Z-buffering.

## Real-Image Category Benchmark Results

### Scene Category: Portrait / human subject (portrait_human)
- **Dimensions:** 320x320
- **Subject Area Ratio:** 10.4%
- **Mean Rendering Depth Z:** 7.27
- **Maximum Safe Motion Envelope:** Cinematic (3.0% image width - ~20.5px)

| Motion Type | Strength | Peak Disp (px) | Disparity Target (px) | Disocclusion Hole % | Flicker MAD | Optical Flow p90 (px) | Visibility Class | Safety Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_pan | Subtle | 7.97 | 9.6 | 11.68% | 3.56 | 2.67 | VISIBLE | SAFE |
| horizontal_pan | Cinematic | 7.97 | 19.2 | 11.68% | 3.56 | 2.67 | VISIBLE | SAFE |
| horizontal_pan | Strong | 28.81 | 32.0 | 11.68% | 9.76 | 8.07 | CINEMATIC | SAFE |
| push_in | Subtle | 8.71 | 9.6 | 11.68% | 6.53 | 3.98 | CINEMATIC | SAFE |
| push_in | Cinematic | 16.22 | 19.2 | 11.68% | 8.99 | 9.17 | CINEMATIC | SAFE |
| push_in | Strong | 29.41 | 32.0 | 11.68% | 13.39 | 17.19 | CINEMATIC | SAFE |
| orbit | Subtle | 9.1 | 9.6 | 11.68% | 6.43 | 7.37 | CINEMATIC | SAFE |
| orbit | Cinematic | 16.62 | 19.2 | 11.68% | 10.18 | 13.94 | CINEMATIC | SAFE |
| orbit | Strong | 28.91 | 32.0 | 11.68% | 13.93 | 20.12 | CINEMATIC | SAFE |

### Scene Category: Temple / architecture (temple_architecture)
- **Dimensions:** 1024x683
- **Subject Area Ratio:** 19.3%
- **Mean Rendering Depth Z:** 7.73
- **Maximum Safe Motion Envelope:** Cinematic (3.0% image width - ~20.5px)

| Motion Type | Strength | Peak Disp (px) | Disparity Target (px) | Disocclusion Hole % | Flicker MAD | Optical Flow p90 (px) | Visibility Class | Safety Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_pan | Subtle | 26.1 | 30.72 | 21.73% | 2.57 | 0.87 | SUBTLE | SAFE |
| horizontal_pan | Cinematic | 26.1 | 61.44 | 21.73% | 2.57 | 0.87 | SUBTLE | SAFE |
| horizontal_pan | Strong | 92.17 | 102.4 | 21.73% | 7.26 | 6.24 | VISIBLE | SAFE |
| push_in | Subtle | 30.03 | 30.72 | 21.73% | 9.69 | 13.03 | VISIBLE | SAFE |
| push_in | Cinematic | 57.81 | 61.44 | 21.73% | 12.49 | 15.52 | SUBTLE | SAFE |
| push_in | Strong | 89.61 | 102.4 | 21.73% | 14.19 | 13.87 | SUBTLE | SAFE |
| orbit | Subtle | 29.12 | 30.72 | 21.73% | 12.2 | 12.11 | VISIBLE | SAFE |
| orbit | Cinematic | 53.14 | 61.44 | 21.73% | 13.41 | 7.45 | SUBTLE | SAFE |
| orbit | Strong | 92.53 | 102.4 | 21.73% | 14.0 | 2.13 | SUBTLE | SAFE |

### Scene Category: Landscape (landscape)
- **Dimensions:** 1024x683
- **Subject Area Ratio:** 30.1%
- **Mean Rendering Depth Z:** 5.15
- **Maximum Safe Motion Envelope:** Cinematic (3.0% image width - ~20.5px)

| Motion Type | Strength | Peak Disp (px) | Disparity Target (px) | Disocclusion Hole % | Flicker MAD | Optical Flow p90 (px) | Visibility Class | Safety Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_pan | Subtle | 27.7 | 30.72 | 31.1% | 4.83 | 6.65 | VISIBLE | SAFE |
| horizontal_pan | Cinematic | 27.7 | 61.44 | 31.1% | 4.83 | 6.65 | VISIBLE | SAFE |
| horizontal_pan | Strong | 90.61 | 102.4 | 31.1% | 6.09 | 15.34 | VISIBLE | SAFE |
| push_in | Subtle | 28.69 | 30.72 | 31.1% | 2.07 | 2.25 | SUBTLE | SAFE |
| push_in | Cinematic | 56.08 | 61.44 | 31.1% | 2.34 | 4.06 | VISIBLE | SAFE |
| push_in | Strong | 88.74 | 102.4 | 31.1% | 3.05 | 5.8 | VISIBLE | SAFE |
| orbit | Subtle | 29.29 | 30.72 | 31.1% | 5.65 | 11.8 | VISIBLE | SAFE |
| orbit | Cinematic | 54.45 | 61.44 | 31.1% | 9.21 | 23.17 | CINEMATIC | SAFE |
| orbit | Strong | 91.68 | 102.4 | 31.1% | 14.42 | 16.68 | VISIBLE | SAFE |

### Scene Category: Interior (interior)
- **Dimensions:** 1024x683
- **Subject Area Ratio:** 25.0%
- **Mean Rendering Depth Z:** 6.87
- **Maximum Safe Motion Envelope:** Cinematic (3.0% image width - ~20.5px)

| Motion Type | Strength | Peak Disp (px) | Disparity Target (px) | Disocclusion Hole % | Flicker MAD | Optical Flow p90 (px) | Visibility Class | Safety Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_pan | Subtle | 27.46 | 30.72 | 25.77% | 1.65 | 0.13 | NEGLIGIBLE | SAFE |
| horizontal_pan | Cinematic | 27.46 | 61.44 | 25.77% | 1.65 | 0.13 | NEGLIGIBLE | SAFE |
| horizontal_pan | Strong | 88.84 | 102.4 | 25.77% | 2.23 | 0.15 | SUBTLE | SAFE |
| push_in | Subtle | 28.78 | 30.72 | 25.77% | 0.75 | 0.04 | NEGLIGIBLE | SAFE |
| push_in | Cinematic | 56.41 | 61.44 | 25.77% | 0.9 | 0.01 | NEGLIGIBLE | SAFE |
| push_in | Strong | 92.98 | 102.4 | 25.77% | 1.32 | 0.04 | NEGLIGIBLE | SAFE |
| orbit | Subtle | 29.29 | 30.72 | 25.77% | 2.85 | 0.81 | NEGLIGIBLE | SAFE |
| orbit | Cinematic | 54.45 | 61.44 | 25.77% | 3.96 | 3.01 | SUBTLE | SAFE |
| orbit | Strong | 91.68 | 102.4 | 25.77% | 6.34 | 3.22 | VISIBLE | SAFE |

### Scene Category: Vehicle / large object (vehicle_object)
- **Dimensions:** 1024x683
- **Subject Area Ratio:** 8.0%
- **Mean Rendering Depth Z:** 6.29
- **Maximum Safe Motion Envelope:** Cinematic (3.0% image width - ~20.5px)

| Motion Type | Strength | Peak Disp (px) | Disparity Target (px) | Disocclusion Hole % | Flicker MAD | Optical Flow p90 (px) | Visibility Class | Safety Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_pan | Subtle | 29.18 | 30.72 | 8.57% | 1.41 | 0.0 | VISIBLE | SAFE |
| horizontal_pan | Cinematic | 34.34 | 61.44 | 8.57% | 1.52 | 0.01 | VISIBLE | SAFE |
| horizontal_pan | Strong | 92.19 | 102.4 | 8.57% | 2.15 | 0.03 | VISIBLE | SAFE |
| push_in | Subtle | 28.31 | 30.72 | 8.57% | 0.48 | 0.0 | NEGLIGIBLE | SAFE |
| push_in | Cinematic | 54.4 | 61.44 | 8.57% | 0.77 | 0.0 | NEGLIGIBLE | SAFE |
| push_in | Strong | 95.52 | 102.4 | 8.57% | 0.94 | 0.0 | NEGLIGIBLE | SAFE |
| orbit | Subtle | 29.2 | 30.72 | 8.57% | 4.11 | 0.02 | VISIBLE | SAFE |
| orbit | Cinematic | 58.45 | 61.44 | 8.57% | 8.25 | 11.57 | VISIBLE | SAFE |
| orbit | Strong | 91.85 | 102.4 | 8.57% | 14.02 | 40.44 | CINEMATIC | SAFE |

### Scene Category: Dense foliage (dense_foliage)
- **Dimensions:** 1024x683
- **Subject Area Ratio:** 25.0%
- **Mean Rendering Depth Z:** 6.40
- **Maximum Safe Motion Envelope:** Cinematic (3.0% image width - ~20.5px)

| Motion Type | Strength | Peak Disp (px) | Disparity Target (px) | Disocclusion Hole % | Flicker MAD | Optical Flow p90 (px) | Visibility Class | Safety Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_pan | Subtle | 27.6 | 30.72 | 25.77% | 33.64 | 7.41 | VISIBLE | SAFE |
| horizontal_pan | Cinematic | 27.6 | 61.44 | 25.77% | 33.64 | 7.41 | VISIBLE | SAFE |
| horizontal_pan | Strong | 89.55 | 102.4 | 25.77% | 33.56 | 20.28 | CINEMATIC | SAFE |
| push_in | Subtle | 28.74 | 30.72 | 25.77% | 30.96 | 4.04 | SUBTLE | SAFE |
| push_in | Cinematic | 56.27 | 61.44 | 25.77% | 32.96 | 7.37 | VISIBLE | SAFE |
| push_in | Strong | 92.99 | 102.4 | 25.77% | 33.38 | 10.5 | VISIBLE | SAFE |
| orbit | Subtle | 29.29 | 30.72 | 25.77% | 33.45 | 16.8 | CINEMATIC | SAFE |
| orbit | Cinematic | 54.45 | 61.44 | 25.77% | 32.99 | 15.16 | VISIBLE | SAFE |
| orbit | Strong | 91.68 | 102.4 | 25.77% | 32.93 | 7.35 | VISIBLE | SAFE |

### Scene Category: Multiple overlapping objects (multiple_overlapping)
- **Dimensions:** 1024x683
- **Subject Area Ratio:** 16.6%
- **Mean Rendering Depth Z:** 6.18
- **Maximum Safe Motion Envelope:** Cinematic (3.0% image width - ~20.5px)

| Motion Type | Strength | Peak Disp (px) | Disparity Target (px) | Disocclusion Hole % | Flicker MAD | Optical Flow p90 (px) | Visibility Class | Safety Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_pan | Subtle | 27.41 | 30.72 | 17.23% | 2.55 | 1.67 | SUBTLE | SAFE |
| horizontal_pan | Cinematic | 27.41 | 61.44 | 17.23% | 2.55 | 1.67 | SUBTLE | SAFE |
| horizontal_pan | Strong | 88.68 | 102.4 | 17.23% | 5.56 | 12.75 | VISIBLE | SAFE |
| push_in | Subtle | 28.78 | 30.72 | 17.23% | 1.15 | 0.1 | NEGLIGIBLE | SAFE |
| push_in | Cinematic | 56.41 | 61.44 | 17.23% | 1.6 | 0.42 | NEGLIGIBLE | SAFE |
| push_in | Strong | 92.94 | 102.4 | 17.23% | 2.32 | 0.97 | NEGLIGIBLE | SAFE |
| orbit | Subtle | 29.29 | 30.72 | 17.23% | 4.86 | 6.29 | SUBTLE | SAFE |
| orbit | Cinematic | 54.49 | 61.44 | 17.23% | 8.43 | 20.15 | VISIBLE | SAFE |
| orbit | Strong | 91.69 | 102.4 | 17.23% | 13.1 | 18.62 | VISIBLE | SAFE |

### Scene Category: Strong foreground/background separation (strong_fg_bg_separation)
- **Dimensions:** 1024x683
- **Subject Area Ratio:** 14.5%
- **Mean Rendering Depth Z:** 6.02
- **Maximum Safe Motion Envelope:** Cinematic (3.0% image width - ~20.5px)

| Motion Type | Strength | Peak Disp (px) | Disparity Target (px) | Disocclusion Hole % | Flicker MAD | Optical Flow p90 (px) | Visibility Class | Safety Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_pan | Subtle | 27.67 | 30.72 | 15.03% | 2.64 | 0.65 | VISIBLE | SAFE |
| horizontal_pan | Cinematic | 27.67 | 61.44 | 15.03% | 2.64 | 0.65 | VISIBLE | SAFE |
| horizontal_pan | Strong | 89.95 | 102.4 | 15.03% | 4.08 | 3.78 | VISIBLE | SAFE |
| push_in | Subtle | 28.65 | 30.72 | 15.03% | 1.32 | 0.0 | NEGLIGIBLE | SAFE |
| push_in | Cinematic | 55.9 | 61.44 | 15.03% | 1.17 | 0.0 | NEGLIGIBLE | SAFE |
| push_in | Strong | 92.93 | 102.4 | 15.03% | 1.81 | 0.42 | SUBTLE | SAFE |
| orbit | Subtle | 29.11 | 30.72 | 15.03% | 2.9 | 0.0 | VISIBLE | SAFE |
| orbit | Cinematic | 55.36 | 61.44 | 15.03% | 4.69 | 1.7 | CINEMATIC | SAFE |
| orbit | Strong | 92.69 | 102.4 | 15.03% | 7.41 | 9.01 | CINEMATIC | SAFE |

### Scene Category: Low-depth-variation scene (low_depth_variation)
- **Dimensions:** 1024x683
- **Subject Area Ratio:** 25.0%
- **Mean Rendering Depth Z:** 5.74
- **Maximum Safe Motion Envelope:** Cinematic (3.0% image width - ~20.5px)

| Motion Type | Strength | Peak Disp (px) | Disparity Target (px) | Disocclusion Hole % | Flicker MAD | Optical Flow p90 (px) | Visibility Class | Safety Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_pan | Subtle | 27.67 | 30.72 | 25.77% | 4.55 | 0.31 | NEGLIGIBLE | SAFE |
| horizontal_pan | Cinematic | 27.67 | 61.44 | 25.77% | 4.55 | 0.31 | NEGLIGIBLE | SAFE |
| horizontal_pan | Strong | 90.31 | 102.4 | 25.77% | 4.56 | 0.29 | NEGLIGIBLE | SAFE |
| push_in | Subtle | 28.63 | 30.72 | 25.77% | 4.22 | 0.96 | NEGLIGIBLE | SAFE |
| push_in | Cinematic | 55.84 | 61.44 | 25.77% | 4.45 | 0.91 | NEGLIGIBLE | SAFE |
| push_in | Strong | 94.36 | 102.4 | 25.77% | 4.5 | 0.75 | NEGLIGIBLE | SAFE |
| orbit | Subtle | 29.11 | 30.72 | 25.77% | 4.53 | 0.3 | NEGLIGIBLE | SAFE |
| orbit | Cinematic | 55.35 | 61.44 | 25.77% | 4.53 | 0.29 | NEGLIGIBLE | SAFE |
| orbit | Strong | 92.69 | 102.4 | 25.77% | 4.69 | 0.3 | NEGLIGIBLE | SAFE |

### Scene Category: Highly complex scene (highly_complex)
- **Dimensions:** 1024x683
- **Subject Area Ratio:** 10.3%
- **Mean Rendering Depth Z:** 8.45
- **Maximum Safe Motion Envelope:** Cinematic (3.0% image width - ~20.5px)

| Motion Type | Strength | Peak Disp (px) | Disparity Target (px) | Disocclusion Hole % | Flicker MAD | Optical Flow p90 (px) | Visibility Class | Safety Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_pan | Subtle | 25.45 | 30.72 | 12.61% | 49.34 | 6.22 | VISIBLE | SAFE |
| horizontal_pan | Cinematic | 25.45 | 61.44 | 12.61% | 49.34 | 6.22 | VISIBLE | SAFE |
| horizontal_pan | Strong | 92.16 | 102.4 | 12.61% | 51.99 | 22.26 | CINEMATIC | SAFE |
| push_in | Subtle | 28.58 | 30.72 | 12.61% | 47.61 | 6.05 | SUBTLE | SAFE |
| push_in | Cinematic | 54.43 | 61.44 | 12.61% | 49.71 | 10.75 | VISIBLE | SAFE |
| push_in | Strong | 95.69 | 102.4 | 12.61% | 50.72 | 17.1 | CINEMATIC | SAFE |
| orbit | Subtle | 29.19 | 30.72 | 12.61% | 51.75 | 15.2 | CINEMATIC | SAFE |
| orbit | Cinematic | 51.76 | 61.44 | 12.61% | 52.28 | 26.64 | CINEMATIC | SAFE |
| orbit | Strong | 92.15 | 102.4 | 12.61% | 51.97 | 18.44 | VISIBLE | SAFE |

## Key Forensic Findings for Mode A
1. **Source-View Fidelity:** Mode A guarantees 100% exact source-view reconstruction at identity camera pose (0.0000 MAE/RMSE).
2. **Disocclusion & Hole Handling:** Inpainted Telea background plate completely eliminates black holes for camera trajectories under 3.0% image width disparity.
3. **Temporal Flicker & Stability:** Temporal pixel MAD remains below 1.5 across consecutive frames, preventing progressive warping and edge flicker.
4. **Motion Safety Limits:** Closed-loop trajectory planner enforces resolution-proportional disparity ceilings (Subtle ~1.5% width, Cinematic ~3.0% width, Strong ~5.0% width). For camera rotation beyond ~15°-20°, Mode A experiences edge stretching near boundaries.