# Phase 3 Quality Honesty Contract

## Core Principles
1. **Never Silently Pretend Quality Was Produced:** If requested resolution or quality cannot be rendered safely, the renderer downgrades explicitly and reports the reason.
2. **No Fake Upscaling:** Final output files upscaled from 720p to 4K are labeled as `INTERPOLATED_UPSCALE` rather than claiming native 4K rendering.
3. **Explicit Device Reporting:** Model execution on CPU is reported strictly as `cpu` without claiming GPU acceleration.
4. **Explicit Mode Distinction:** Mode A is reported as `2.5d_layered_splatting` and Mode B as `3d_explicit_reconstruction`.

## Quality Report Schema (`quality_report.json`)
```json
{
  "quality_honesty_contract_version": "1.0",
  "REQUESTED": {
    "render_mode": "auto",
    "resolution": "1080p",
    "quality": "high",
    "hardware": "cpu",
    "motion": "Cinematic Push-In",
    "requested_frames": 48
  },
  "ACTUAL": {
    "selected_mode": "2.5d",
    "actual_output_resolution": "1280x720",
    "actual_quality_profile": "CPU_QUALITY",
    "actual_hardware": "cpu",
    "actual_model_device": "cpu",
    "actual_frames": 48,
    "actual_render_time_sec": 8.41,
    "peak_ram_gb": 1.18,
    "peak_vram_gb": 0.0
  },
  "LIMITATIONS": {
    "cpu_ceiling_enforced": true,
    "downgrade_warning": "CPU execution constrained: requested 1080p downgraded to 720p ceiling for resource safety.",
    "max_safe_viewpoint_angle_deg": 15,
    "disocclusion_hole_pct": 11.68,
    "reconstruction_confidence": 0.92,
    "4k_support_status": "UNSUPPORTED_ON_CPU"
  }
}
```