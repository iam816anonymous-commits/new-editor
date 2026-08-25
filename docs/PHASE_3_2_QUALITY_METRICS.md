# Phase 3.2 Quality Metrics & Validation Schema

## 1. Automated Parallax Quality Metrics
- **Foreground Displacement ($\text{FG}_{\text{disp}}$):** Mean pixel displacement inside subject mask.
- **Background Displacement ($\text{BG}_{\text{disp}}$):** Mean pixel displacement in background.
- **Relative Parallax Shift:** $\Delta_{\text{parallax}} = \text{FG}_{\text{disp}} - \text{BG}_{\text{disp}}$.
- **Subject Scale Growth:** Ratio of synthesized subject area growth.
- **Motion Visibility Classification:** `WEAK`, `SUBTLE`, `VISIBLE`, `CINEMATIC`, `EXCESSIVE`.
- **4-Tier Pass Gate:** `MATHEMATICAL_PASS`, `RASTER_PASS`, `PERCEPTUAL_PASS`, `FINAL_PASS`.
