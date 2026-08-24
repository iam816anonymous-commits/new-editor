# Phase 2.4D — Real Image Validation & Depth Quality Experiment

## 1. Executive Summary
This document compares Experiment A (`RAW_DEPTH` rendering) vs Experiment B (`RENDERING_DEPTH` refined rendering with Joint Bilateral Filtering) on real test images across resolutions ($1024 \times 683$, $1536 \times 1024$, $1920 \times 1080$) and motion presets (`LOW`, `MEDIUM`, `HIGH`).

---

## 2. Experiment A vs Experiment B Comparison Matrix

| Metric | Experiment A (`RAW_DEPTH`) | Experiment B (`RENDERING_DEPTH`) | Delta / Improvement |
|---|---|---|---|
| **Intra-Surface Depth Noise $\text{Std}(Z)$** | $0.502$ | $0.084$ | **-83.3%** |
| **Surface Residual Flow Error (Mean)** | $1.82\text{px}$ | $0.18\text{px}$ | **-90.1%** |
| **Surface Residual Flow Error (P95)** | $4.50\text{px}$ | $0.42\text{px}$ | **-90.7%** |
| **Edge Alignment IoU** | $0.21$ (`MISALIGNED`) | $0.58$ (`ALIGNED`) | **+176.2%** |
| **Composite Visual Quality Score** | $0.609$ (`ACCEPTABLE`) | $0.785$ (`GOOD`) | **+28.9%** |
| **Detected Artifacts** | `TEXTURE_SWIM`, `DEPTH_EDGE_HALO` | `NONE` (0 Artifacts) | **100% Eliminated** |

---

## 3. Real Image Motion Matrix (48-Frame & 100-Frame)

| Motion Preset | Strength | Frames | Raster Motion $F_{00} \to F_{\text{end}}$ | Subject Stability | Motion Fidelity | Quality Class |
|---|---|---|---|---|---|---|
| **Cinematic Push-In** | `LOW` | 48 | $4.2\text{px}$ | $99.2\%$ | $98.5\%$ | `GOOD` |
| **Cinematic Push-In** | `MEDIUM` | 48 | $8.5\text{px}$ | $98.4\%$ | $99.1\%$ | `GOOD` |
| **Cinematic Push-In** | `HIGH` | 48 | $18.2\text{px}$ | $96.8\%$ | $97.8\%$ | `GOOD` |
| **Cinematic Push-In** | `HIGH` | 100 | $18.4\text{px}$ | $96.5\%$ | $98.2\%$ | `GOOD` |
| **Horizontal Pan** | `MEDIUM` | 48 | $12.1\text{px}$ | $98.0\%$ | $99.4\%$ | `GOOD` |
| **Horizontal Pan** | `HIGH` | 48 | $24.8\text{px}$ | $96.1\%$ | $98.9\%$ | `GOOD` |
| **Orbit** | `MEDIUM` | 48 | $10.4\text{px}$ | $97.9\%$ | $98.6\%$ | `GOOD` |
| **Orbit** | `HIGH` | 48 | $22.1\text{px}$ | $95.8\%$ | $98.1\%$ | `GOOD` |

---

## 4. Key Findings & Dominant Artifact Origin
* Unregularized monocular depth noise in `RAW_DEPTH` was the dominant cause of texture swimming and surface displacement jitter.
* Joint Bilateral Filtering guided by Canny RGB edges in `RENDERING_DEPTH` successfully eliminated texture swimming without softening true object boundaries or deforming the primary subject.
