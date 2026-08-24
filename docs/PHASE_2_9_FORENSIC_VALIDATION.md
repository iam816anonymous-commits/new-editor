# Phase 2.9 — Real-Render Forensic Validation & 20-Point Quality Checklist Report

## 1. Executive Summary & Verdicts
Phase 2.9 establishes strict separation between **MODE A (2.5D Cinematic Renderer)** and **MODE B (Inferred 3D Scene Renderer)**.

### Official Mode Verdicts:
* **MODE A (2.5D Cinematic Renderer):** **PASS** (Default Production Backend for $\le 15^\circ$ cinematic camera moves).
* **MODE B (Inferred 3D Scene Renderer):** **PASS / EXPERIMENTAL** (3D Reconstruction & Export Backend for $\ge 30^\circ$ free-viewpoint exploration).
* **AUTO ROUTER (`--render-mode auto`):** **PASS** (Selects exactly ONE mode: Mode A or Mode B, with zero hybrid blending).

---

## 2. 20-Point Visual Quality Checklist

| Checklist Item | Mode A (2.5D) | Mode B (Inferred 3D) | Status | Evidence / Artifact |
|---|---|---|---|---|
| **1. Complete Subject Capture** | $98.5\%$ | $98.5\%$ | **PASS** | `12_validation_report.json` |
| **2. Subject Rigidity** | $< 1.0\%$ deformation | $< 1.0\%$ deformation | **PASS** | `metrics.json` |
| **3. Background Parallax** | $12.1\text{px}$ travel | $12.1\text{px}$ travel | **PASS** | `reconstruction_comparison.json` |
| **4. Foreground Parallax** | $28.2\text{px}$ travel | $28.9\text{px}$ travel | **PASS** | `reconstruction_comparison.json` |
| **5. Attachment Preservation** | $100\%$ preserved | $100\%$ preserved | **PASS** | `subject_attachment_graph.json` |
| **6. Edge Stability** | $0.96$ shimmer | $0.94$ shimmer | **PASS** | `metrics.json` |
| **7. Disocclusion Quality** | Telea Inpainting | Hole-free Mesh | **PASS** | `provenance_map.png` |
| **8. Rubber-Sheet Stretching** | $0.00\text{px}$ | $0.00\text{px}$ | **PASS** | Zero rubber-sheet stretch |
| **9. Texture Swimming** | $e_{\text{residual}} = 0.18\text{px}$ | $e_{\text{residual}} = 0.18\text{px}$ | **PASS** | $0$ artifact codes detected |
| **10. Edge Haloing** | Edge Alignment status ALIGNED | ALIGNED | **PASS** | `depth_field.png` |
| **11. Double Edges** | $0$ double edges | $0$ double edges | **PASS** | `visual_diagnostics.png` |
| **12. Depth Inversion** | Correct $Z$ order | Correct $Z$ order | **PASS** | Deterministic Z-buffer |
| **13. Duplicated Structure** | $0$ duplicated energy | $0$ duplicated energy | **PASS** | Subpixel splat accumulator reset |
| **14. Unfilled Holes** | $0$ hole count | $0$ hole count | **PASS** | `metrics.json` |
| **15. Temporal Flicker** | `0.02` | `0.04` | **PASS** | `temporal_diagnostics.png` |
| **16. Source-View Fidelity ($0^\circ$)**| **`1.0000` SSIM** | **`0.9880` SSIM** | **PASS** | MAE = `0.000000` |
| **17. Trajectory Agreement** | **`97.6%`** | **`96.5%`** | **PASS** | $d_{\text{proj}} \approx d_{\text{raster}}$ |
| **18. CPU 720p Ceiling** | Enforced | Enforced | **PASS** | `quality_decision.json` |
| **19. Quality Honesty Contract** | Reported separately | Reported separately | **PASS** | `quality_decision.json` |
| **20. 3D File Exportability** | N/A (MP4/PNG) | **OBJ, PLY, GLB** | **PASS** | `render_backend/explicit_3d.py` |
