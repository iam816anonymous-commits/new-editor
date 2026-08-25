# Phase 3.2 Disocclusion & Inpainting Engine

## 1. Disocclusion Processing
Camera translation reveals unobserved background pixels behind foreground objects.
1. **Background Plate Generation:** Conservative morphology dilation creates a hole mask over the subject region.
2. **Inpainting:** Uses Navier-Stokes / Telea boundary propagation (`cv2.inpaint`) on RGB and depth arrays.
3. **Observed Pixel Preservation:** Observed background pixels outside the subject mask are preserved with 100% fidelity ($P(x,y) = 1.0$).

## 2. Disocclusion Metrics
- `reconstructed_pixel_percentage`: Percentage of output frame pixels synthesised via inpainting.
- `boundary_risk_exposure`: Mean exposure risk in complex silhouette boundary zones.
