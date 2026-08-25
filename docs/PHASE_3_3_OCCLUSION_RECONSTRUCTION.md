# Phase 3.3 — Occlusion Boundary Detection & Disocclusion Forecasting

## Executive Summary
Occlusion boundaries represent severe depth discontinuities where background pixels are hidden behind foreground objects in the initial view frame. Under camera movement, these regions become disoccluded, exposing unobserved spatial voids ("disocclusion holes"). Phase 3.3 provides explicit occlusion boundary detection and forecasting.

## Algorithmic Method
Located in `modes/mode_2_5d/scene.py` via `detect_occlusion_boundaries`:
1. **Depth Gradient Analysis**: Sobel operator $\nabla Z = (\frac{\partial Z}{\partial x}, \frac{\partial Z}{\partial y})$ measures local depth variation across the continuous depth field.
2. **Normalized Thresholding**: Occlusion edges are extracted using range-normalized gradient thresholds:
   $$\text{Edges} = \|\nabla Z\| > \tau \cdot \text{ptp}(Z)$$
   where $\tau = 0.08$ (8% of depth dynamic range).
3. **Disocclusion Exposure Forecasting**: Boundaries are dilated along motion trajectory vectors to forecast missing pixel regions prior to frame synthesis.

## Boundary Risk Map
Disocclusion risk areas are exported in `boundary_risk_map.png` and recorded in diagnostic summaries to guide closed-loop trajectory amplitude planning in `camera/safety.py`.
