# Phase 3 Hardware Quality Matrix

## Executive Summary
This document establishes the empirical hardware quality matrix for the renderer across CPU and GPU hardware tiers.

## Hardware Profiles & Execution Budgets

| Profile ID | Device Class | VRAM (GB) | RAM (GB) | CPU Cores | Max Honest Resolution | CPU Ceiling Enforced | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU_LOW_MEMORY | CPU (Low RAM) | 0.0 | 8.0 | 2 | 480p (854x480) | True | SUPPORTED |
| CPU_STANDARD | CPU (Standard 16GB) | 0.0 | 16.0 | 4 | 720p (1280x720) | True | SUPPORTED |
| GPU_LOW_VRAM | NVIDIA GTX 1650 / RTX 3050 (4GB VRAM) | 4.0 | 16.0 | 8 | 720p (1280x720) | False | SUPPORTED |
| GPU_STANDARD | NVIDIA RTX 3060 / RTX 4060 (8GB VRAM) | 8.0 | 32.0 | 12 | 1080p (1920x1080) | False | SUPPORTED |
| GPU_HIGH_VRAM | NVIDIA RTX 3090 / RTX 4090 (24GB VRAM) | 24.0 | 64.0 | 16 | 1440p / 4K (3840x2160) | False | CONDITIONALLY_SUPPORTED |

## CPU Execution Contract
- **Hard Ceiling:** CPU mode is strictly capped at 720p (1280x720).
- **Downgrade Warning:** If a user requests 1080p, 1440p, or 4K under CPU mode, the pipeline explicitly downgrades to 720p with a Quality Honesty warning in `quality_report.json`.
- **Memory Budget:** Peak RAM footprint under CPU 720p remains bounded at ~1.2 GB.

## 4K Support Feasibility Status
- **CPU:** `UNSUPPORTED` (4K rendering on CPU violates runtime and memory safety bounds).
- **GPU <= 8GB VRAM:** `UNSUPPORTED` (VRAM thrashing occurs above 1080p).
- **GPU >= 16GB-24GB VRAM:** `CONDITIONALLY_SUPPORTED` (Requires native 4K depth estimation and >= 16GB VRAM; final upscaling from 1080p is explicitly reported as interpolated).