# Phase 3.3 Forensic Baseline Inspection Report

## 1. Current Renderer Inspection & Capabilities Audit

| Metric / Capability | Baseline State | Target Neural Layered State |
|---------------------|----------------|-----------------------------|
| Scene Representation | Per-pixel monocular depth array | Persistent `LayeredScene` with 4 adaptive layers (`BACKGROUND`, `MIDGROUND`, `PRIMARY_SUBJECT`, `FOREGROUND`) |
| Occlusion Boundaries | Implicit depth Sobel edges | Explicit occlusion boundary detection & hidden-region masking |
| Hidden RGB Reconstruction | Per-frame Telea inpainting on missing splats | Pre-animation persistent background reconstruction |
| Layer Processing | Dual-pass (Foreground vs Background) | Multi-layer independent depth transformation & alpha compositing |
| Camera Pivot | Fixed principal point | Adaptive convergence depth & pivot near primary subject depth |
| Canvas Overscan | Standard resolution | Overscanned internal canvas with safe crop region |

## 2. Baseline Architecture Audit
- **Layer Extraction:** Currently relies on `subject_mask` and `bg_plate`.
- **Pre-Animation Inpainting:** Currently `reconstruct_background_rgb` fills subject holes on the background plate before animation.
- **Z-Buffer Compositing:** Foreground layer strictly overwrites background layer where valid foreground splats exist.
