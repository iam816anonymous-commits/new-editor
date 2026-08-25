# Phase 3.2 Baseline Forensic Inspection Report

## 1. Baseline Pipeline Architecture & Data Flow

```
Input Image (RGB Array)
  │
  ├─► Depth Anything V2 Small ──► RAW_DEPTH
  │                                   │
  │                                   ▼
  │                             Percentile Outlier Clipping (p1=1%, p99=99%)
  │                                   │
  │                                   ▼
  │                             Rendering Depth Z in [0.1, 10.0]
  │                                   │
  ├─► SAM 2 Hiera-Tiny ─────────► Subject Mask
  │                                   │
  │                                   ▼
  ├─► Bilateral Edge Refinement ──► REFINED_DEPTH
  │                                   │
  ├─► Disocclusion Inpainting ────► Clean Background Plate & BG Depth
  │                                   │
  └─► Closed-Loop Trajectory ─────► Poses (R_k, t_k)
        Safety Planner                │
                                      ▼
                               Forward Subpixel Splatting Engine:
                               P' = P @ R^T + t_k * m_layer
                               u' = fx * X' / Z' + cx
                               v' = fy * Y' / Z' + cy
```

## 2. Quantitative Baseline Measurements

1. **Depth Normalization Range:** Normalized scene depth $Z \in [0.1, 10.0]$. High disparity (foreground) maps to $Z \approx 0.1$, low disparity (far background) maps to $Z \approx 10.0$.
2. **Camera-Relative Image Displacement:**
   - For pure horizontal translation $t_x$, pixel displacement $\Delta u = f_x \cdot t_x / Z$.
   - Foreground ($Z = 1.0$): $\Delta u = 320 \cdot 0.05 / 1.0 = 16.0$ pixels.
   - Background ($Z = 8.0$): $\Delta u = 320 \cdot 0.05 / 8.0 = 2.0$ pixels.
   - Foreground / Background Motion Ratio: $16.0 / 2.0 = 8.0\times$ differential parallax shift.
3. **Motion Strength Scaling Tiers:**
   - `subtle`: Disparity ceiling $3.0\%$ image dimension.
   - `cinematic` / `balanced`: Disparity ceiling $6.0\%$ image dimension.
   - `strong`: Disparity ceiling $10.0\%$ image dimension.
4. **Subpixel Forward Splatting:** Bilinear distribution to 2x2 target pixel neighborhood with deterministic Z-buffering (closer camera-space Z overwrites far surfaces).
5. **Disocclusion Handling:** Inpaints newly exposed background holes using Telea / Navier-Stokes boundary propagation on background plate while strictly preserving observed pixels.
6. **Temporal Stability:** Frame sequence generated via continuous $C^1$ smoothstep easing $s(t) = 6t^5 - 15t^4 + 10t^3$ with zero frame-to-frame velocity jumps.
