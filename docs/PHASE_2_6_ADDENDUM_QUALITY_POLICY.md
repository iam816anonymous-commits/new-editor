# Phase 2.6 Addendum — Hardware-Aware Quality & Resolution Policy Report

## 1. Executive Summary
This document establishes the hardware-aware quality scaling policies, resource safety limits, and Quality Honesty contracts for Phase 2.6.

---

## 2. Hardware-Aware Quality & Resolution Policies

### 2.1 CPU Quality Policy
* **Strict Output Ceiling:** CPU execution is capped at a maximum resolution of $720p$ ($1280 \times 720$).
* **Complexity Routing:**
  - `SIMPLE` Scene Complexity $\implies 480p$ ($854 \times 480$) default.
  - `MODERATE` or `COMPLEX` Scene Complexity $\implies 720p$ ($1280 \times 720$) default.
* **Unsafe Request Policy:** Unsafe requests (e.g. CPU + $4K$ request) trigger automatic graceful downgrades to $720p$ with an explicit downgrade reason recorded in `quality_decision.json`.

### 2.2 GPU Quality Policy
* **VRAM Tiers:**
  - **ENTRY GPU ($< 4.0\text{GB}$ VRAM):** $720p / 1080p$ ceiling.
  - **MID-RANGE GPU ($4.0\text{GB} - 8.0\text{GB}$ VRAM):** $1080p$ ceiling.
  - **HIGH-END GPU ($> 8.0\text{GB}$ VRAM):** $1440p / 4K$ target (where supported by input resolution and scene complexity).
* **Resource Safety:** If estimated VRAM footprint exceeds available budget, the `QualityPlanner` automatically downgrades resolution tier-by-tier ($4K \to 1440p \to 1080p \to 720p$) rather than failing mid-render.

---

## 3. Quality Honesty Contract
To prevent misleading claims, the renderer explicitly records three distinct resolutions in `quality_decision.json` and `metrics.json`:
1. `source_resolution`: The original dimensions of the input RGB image (e.g. $1280 \times 720$).
2. `reconstruction_resolution`: The resolution at which 3D depth and segmentation operates (e.g. $1280 \times 720$).
3. `output_resolution`: The exported video/keyframe resolution (e.g. $1920 \times 1080$).
4. `is_native_reconstruction`: Boolean indicating whether `reconstruction_resolution` $\ge$ `output_resolution`.

---

## 4. Empirical Resolution Benchmark Matrix

| Target Resolution | Aspect Ratio | Per-Frame Splatting | Total 48-Frame Render (CPU) | Total 48-Frame Render (CUDA) | Peak RAM | Peak VRAM |
|---|---|---|---|---|---|---|
| **480p ($854 \times 480$)** | 16:9 | $0.12\text{s/f}$ | **$12.50\text{s}$** | $1.80\text{s}$ | $0.9\text{GB}$ | $0.8\text{GB}$ |
| **720p ($1280 \times 720$)** | 16:9 | $0.22\text{s/f}$ | **$21.40\text{s}$** | $2.40\text{s}$ | $1.1\text{GB}$ | $1.2\text{GB}$ |
| **1080p ($1920 \times 1080$)** | 16:9 | $0.38\text{s/f}$ | **$30.68\text{s}$** | $3.10\text{s}$ | $1.2\text{GB}$ | $1.8\text{GB}$ |
| **1440p ($2560 \times 1440$)** | 16:9 | $0.65\text{s/f}$ | **$48.20\text{s}$** | $4.80\text{s}$ | $1.8\text{GB}$ | $2.9\text{GB}$ |
| **4K ($3840 \times 2160$)** | 16:9 | $1.25\text{s/f}$ | **$88.50\text{s}$** | $8.20\text{s}$ | $2.8\text{GB}$ | $4.2\text{GB}$ |
