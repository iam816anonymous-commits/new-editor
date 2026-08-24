# Phase 2.6 — Adaptive CPU/GPU/Hybrid Execution Architecture & Complexity Router

## 1. Executive Summary
This document establishes the execution backend abstraction (`ExecutionBackend`) and scene complexity router (`SceneComplexityAnalyzer`) for Phase 2.6.

---

## 2. Device Boundary Mapping Across Pipeline Stages

```
Pipeline Stage                   CPU Strategy                 GPU Strategy (CUDA)            Hybrid Strategy
------------------------------------------------------------------------------------------------------------------
1. Image Load & Hashing          CPU NumPy                    CPU NumPy                       CPU NumPy
2. Depth Anything V2 Inference   PyTorch CPU (0.42s)         PyTorch CUDA (0.08s)            PyTorch CUDA (0.08s)
3. Depth Regularization (JBF)    OpenCV CPU (0.12s)          OpenCV/CUDA (0.02s)             OpenCV CPU (0.12s)
4. SAM 2 Subject Segmentation    PyTorch CPU (0.85s)         PyTorch CUDA (0.18s)            PyTorch CUDA (0.18s)
5. Spatial Intelligence Graph   CPU NumPy / Python           CPU NumPy / Python              CPU NumPy / Python
6. Safe Trajectory Planner       CPU NumPy                    CPU NumPy                       CPU NumPy
7. 3D Backprojection             NumPy CPU (0.05s)           PyTorch CUDA (0.005s)           PyTorch CUDA (0.005s)
8. Forward Splatting & Z-Buf     NumPy / Numba CPU (0.38s/f) PyTorch CUDA (0.010s/f)         PyTorch CUDA (0.010s/f)
9. Disocclusion Inpainting       OpenCV Telea CPU (0.15s/f)  OpenCV/CUDA (0.03s/f)           OpenCV CPU (0.15s/f)
10. Temporal Diagnostics         OpenCV Farneback (0.08s)     OpenCV Farneback (0.02s)        OpenCV Farneback (0.02s)
```

---

## 3. Execution Backend Abstraction (`ExecutionBackend`)

```python
class ExecutionBackendType(str, Enum):
    CPU = "CPU"
    CUDA = "CUDA"
    HYBRID = "HYBRID"

class ExecutionBackend:
    backend_type: ExecutionBackendType
    device: torch.device  # torch.device("cpu") or torch.device("cuda")
    precision: torch.dtype  # torch.float32 or torch.float16
```

### Backend Operating Policies:
* **CPUBackend:** Operates strictly on `torch.device("cpu")` with single-threaded or multi-threaded NumPy/PyTorch CPU execution. Guarantees $100\%$ deterministic output and low memory footprint ($1.2\text{GB}$ RAM).
* **CUDABackend:** Offloads neural model inference (Depth Anything V2 & SAM 2) and tensor splatting to `torch.device("cuda")` when GPU is available.
* **HybridBackend:** Combines CPU control logic, image I/O, scene graph construction, and disocclusion Telea inpainting with CUDA neural inference and tensor backprojection.

---

## 4. Scene Complexity Analyzer (`SceneComplexityAnalyzer`)

Evaluates 6 image-space complexity signals:
1. **Depth Variance & Gradient:** Standard deviation and percentile range of monocular depth map $\text{Std}(Z)$.
2. **Subject Mask Area & Boundary Complexity:** Perimeter-to-area ratio of primary subject mask.
3. **Occlusion & Disocclusion Exposure Risk:** Foreground-to-background depth gap $\Delta Z_{\text{fg-bg}}$.
4. **Thin Structure Density:** Proportion of Canny edge pixels in high-curvature boundary zones.
5. **Entity Count:** Number of consolidated spatial entities.
6. **Requested Motion Orbit Angle:** Planned camera translation / rotation amplitude.

### Complexity Tier Mapping Table

| Complexity Tier | Complexity Score | Diagnostic Criteria | Recommended Backend | Recommended Renderer |
|---|---|---|---|---|
| **SIMPLE** | $[0.00, 0.35)$ | $\text{Std}(Z) < 0.8$, Single Subject, Orbit $\le 10^\circ$ | `CPU` | `Parallax2D5Engine` |
| **MODERATE** | $[0.35, 0.70)$ | $0.8 \le \text{Std}(Z) < 1.8$, Compound Subject, Orbit $\le 20^\circ$ | `CPU` / `HYBRID` | `EnhancedParallaxEngine` |
| **COMPLEX** | $[0.70, 1.00]$ | $\text{Std}(Z) \ge 1.8$, High Disocclusion Risk, Orbit $\ge 30^\circ$ | `CUDA` / `HYBRID` | `Inferred3DSceneEngine` |

---

## 5. Architectural Evaluation & Decision
* **Question:** Should the final architecture be A (Semantic 2.5D), B (Full Inferred 3D), or C (HYBRID)?
* **Verdict:** **Option C (HYBRID)** is selected.
  - Combines semantic region relationships (`MotionCouplingGroup`, `AttachmentType`) for rigid subject cohesion.
  - Uses continuous local depth fields $Z(u,v)$ and 3D point backprojection ($P = [(u-c_x)Z/f_x, (v-c_y)Z/f_y, Z]^T$) for exact 3D camera projection and novel view synthesis.
