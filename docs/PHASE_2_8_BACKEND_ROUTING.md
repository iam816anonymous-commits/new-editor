# Phase 2.8 — Confidence-Aware Adaptive Backend Router & Quality Honesty Report

## 1. Executive Summary
This document defines the confidence-aware adaptive backend router and Quality Honesty contract for selecting between Mode A (2.5D), Mode B (HYBRID), and Mode C (Inferred 3D).

---

## 2. Adaptive Backend Routing Decision Matrix

```
INPUT IMAGE + CAMERA MOTION
            ↓
  [SceneComplexityAnalyzer]
            ↓
  [HardwareProfile Detect]
            ↓
  [QualityPlanner Routing]
            ↓
 ┌──────────────────────────────────────────────────┐
 │                                                  │
 ↓                                                  ↓
Simple Scene + CPU + <=15° Orbit          Complex Scene + GPU + >=30° Orbit
      ↓                                                  ↓
Mode B (HYBRID Reference Backend)          Mode C (Inferred 3D Export Backend)
```

| Input Parameters & Conditions | Routing Confidence | Selected Backend | Selected Quality Profile | Rationale |
|---|---|---|---|---|
| **CPU + SIMPLE + Orbit $\le 10^\circ$** | `0.98` | **Mode B (HYBRID)** | `CPU_FAST` ($480p$) | Low memory footprint ($0.9\text{GB}$), $100\%$ source view fidelity |
| **CPU + MODERATE + Orbit $\le 15^\circ$** | `0.95` | **Mode B (HYBRID)** | `CPU_QUALITY` ($720p$) | CPU $720p$ ceiling enforced for resource safety |
| **GPU + MODERATE + Orbit $\le 20^\circ$** | `0.94` | **Mode B (HYBRID)** | `GPU_BALANCED` ($1080p$) | Optimal cinematic camera motion quality and temporal stability |
| **GPU + COMPLEX + Orbit $\ge 30^\circ$** | `0.92` | **Mode C (INFERRED 3D)** | `GPU_HIGH` ($1440p / 4K$) | Explicit 3D mesh prevents rubber-sheet stretching at extreme angles |
| **Low Confidence Geometry / Unsafe 4K** | `0.65` | **Mode B Fallback** | `GPU_BALANCED` ($1080p$) | Automatic VRAM safety downgrade to $1080p$ |

---

## 3. Quality Honesty Contract
To ensure quality parameters are transparent and accurate:
* `source_resolution`: Original image input dimensions (e.g. $1536 \times 1024$).
* `reconstruction_resolution`: Depth & segmentation operating resolution (e.g. $1536 \times 1024$).
* `output_resolution`: Exported keyframe/video resolution (e.g. $1920 \times 1080$).
* `is_native_reconstruction`: `True` when reconstruction resolution $\ge$ output resolution.
* `quality_decision.json` and `backend_decision.json` export decision logs for every render job.
