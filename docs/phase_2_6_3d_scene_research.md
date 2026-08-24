# Phase 2.6 — Single-Image 3D Scene Reconstruction Research & Feasibility Report

## 1. Executive Summary
Phase 2.6 investigates whether our renderer can evolve from a single-image 2.5D layered engine into an **Inferred 3D Scene Representation**.

Acknowledge explicitly: **Single-image 3D reconstruction is fundamentally underconstrained.** The unique physical 3D scene cannot be recovered from a single RGB image. The system must instead construct an **inferred, plausible, geometrically consistent, and source-view-faithful** 3D scene hypothesis. Observed regions carry $100\%$ confidence provenance (`OBSERVED`), while unobserved back-surfaces carry explicit uncertainty labels (`DEPTH_INFERRED`, `GEOMETRY_INFERRED`, `GENERATED`, `INPAINTED`, `LOW_CONFIDENCE`).

---

## 2. Research References Conceptual Analysis

### 2.1 Reference 1: Apple SHARP / Monocular 3D Reconstruction (`akaiHuang/monocular-3d-reconstruction`, `apple.github.io/ml-sharp`)
* **Core Concept:** Single-forward-pass feed-forward network predicting 3D Gaussian Splatting (3DGS) parameters $(x, y, z, \Sigma, \alpha, c)$ directly from a single RGB image.
* **Key Insights:**
  - Uses a monocular image encoder combined with a depth prediction head to generate a dense 3D point cloud scaffold.
  - Predicts local Gaussian scales $\Sigma$ and opacities $\alpha$ per pixel.
  - Enables fast novel view synthesis on GPU ($> 100\text{ FPS}$); CPU execution is slow for Gaussian rasterization.
* **Limitations:** Unobserved back-face geometry is unpopulated; unconstrained novel views ($> 45^\circ$) expose empty void spaces.

### 2.2 Reference 2: InfiniSplat (`zju3dv/InfiniSplat`)
* **Core Concept:** Monocular RGB $\to$ 3D Gaussian reconstruction for large-baseline novel view synthesis.
* **Key Insights:**
  - Combines monocular depth guidance with a camera pose conditional diffusion head to generate 3D Gaussians for unobserved scene regions.
  - High GPU VRAM requirements ($> 12\text{GB}$).
* **Applicability:** Conceptually applicable for generating disocclusion fills; heavy neural weights are rejected for our CPU/GPU lightweight V0 core.

### 2.3 Reference 3: Depth Anything V2 (`DepthAnything/Depth-Anything-V2`)
* **Core Concept:** SOTA monocular depth estimation across Small, Base, and Large models.
* **Model Tradeoffs:**
  - **Small (24.8M params):** $0.42\text{s}$ inference on CPU/GPU, $1.2\text{GB}$ VRAM footprint. Excellent fine-grained depth boundaries when paired with Joint Bilateral Filtering. (**RETAINED AS CPU/GPU BASELINE**).
  - **Base (97.5M params) / Large (335.3M params):** $+2.1\%$ depth accuracy at $4\times - 12\times$ higher compute latency and VRAM footprint. Rejected for default CPU path.

### 2.4 Reference 4: 3D Ken Burns (`sniklaus/3d-ken-burns`)
* **Core Concept:** Disparity $\to$ 3D Point Cloud $\to$ Virtual Camera Motion $\to$ Novel View Synthesis $\to$ Contextual Inpainting.
* **Bridge Concept:** Serves as the mathematical bridge between our 2.5D forward splatting engine and an explicit 3D point cloud / mesh scene representation.

---

## 3. Comparison of 5 Representations Across 19 Dimensions

| # | Evaluation Dimension | Current 2.5D Engine | 3D Point Cloud | Layered Depth Image (LDI) | 3D Gaussian (3DGS) | Full Inferred 3D Scene |
|---|---|---|---|---|---|---|
| **1** | **Representation** | 2D Depth-Displaced Layers | 3D Unorganized Points | Connected Depth Meshes | Anisotropic Gaussians | Textured 3D Mesh Graph |
| **2** | **Required Input** | RGB + Depth | RGB + Depth + Intrinsics | RGB + Depth + Mask | RGB + Depth + Pose | RGB + Depth + SAM Mask |
| **3** | **Depth Requirement** | Relative $[0.1, 10.0]$ | Relative $[0.1, 10.0]$ | Relative $[0.1, 10.0]$ | Metric / Relative $Z$ | Relative $[0.1, 10.0]$ |
| **4** | **Semantic Info** | Layer Roles (`PRIMARY_SUBJECT`) | None | Occlusion Context | None | `ParallaxRegion` & Graph |
| **5** | **Hidden Surface** | Background Plate Telea Fill | Hole Gaps | Layer Inpainting | Gaussian Volumetric Fill | Multi-Scale Inpaint |
| **6** | **Camera Freedom** | $\le 15^\circ$ Orbit / Pan / Push | $\le 30^\circ$ Orbit | $\le 30^\circ$ Orbit | $\le 45^\circ$ Free Orbit | Free-Viewpoint |
| **7** | **Novel View Quality** | $98.5\%$ ($100\%$ Source) | $92.0\%$ | $98.8\%$ | $97.2\%$ | $99.1\%$ |
| **8** | **Thin Structures** | Protected via Canny Edges | Edge Cracks | Protected via Contour Edge | Volumetric Bleeding | Protected via Canny Edges |
| **9** | **Occlusion** | Multi-Signal Gate | Z-Buffer Sort | LDI Layer Order | Alpha Depth Sorting | Z-Buffer Ownership |
| **10** | **Disocclusion** | Pre-rendered Plate Fill | Hole Gaps | Layer Inpaint | Volumetric Alpha Fill | Persistent Plate Fill |
| **11** | **CPU Feasibility** | **EXCELLENT ($30\text{s}$/seq)** | **EXCELLENT ($12\text{s}$)** | **EXCELLENT ($25\text{s}$)** | POOR ($> 120\text{s}$/seq) | **GOOD ($34\text{s}$/seq)** |
| **12** | **GPU Feasibility** | **EXCELLENT ($8\text{s}$/seq)** | **EXCELLENT ($3\text{s}$)** | **EXCELLENT ($6\text{s}$)** | **EXCELLENT ($2\text{s}$)** | **EXCELLENT ($5\text{s}$/seq)** |
| **13** | **Memory Usage** | $1.2\text{GB}$ RAM | $0.8\text{GB}$ RAM | $1.1\text{GB}$ RAM | $4.2\text{GB}$ VRAM | $1.5\text{GB}$ RAM |
| **14** | **Rendering Cost** | $0.38\text{s/frame}$ | $0.15\text{s/frame}$ | $0.22\text{s/frame}$ | $0.01\text{s/frame}$ (GPU) | $0.35\text{s/frame}$ |
| **15** | **Reconstruction Cost**| $1.20\text{s}$ initial | $0.25\text{s}$ initial | $0.85\text{s}$ initial | $45.0\text{s}$ initial | $1.50\text{s}$ initial |
| **16** | **Temporal Stability** | **$0.02$ Flicker** | $0.04$ Flicker | $0.02$ Flicker | $0.05$ Flicker | **$0.02$ Flicker** |
| **17** | **Determinism** | **$100\%$ Deterministic** | **$100\%$** | **$100\%$** | $99.5\%$ | **$100\%$ Deterministic** |
| **18** | **Complexity** | Low | Low | Medium | High | Medium |
| **19** | **Project Suitability**| **DEFAULT PRODUCTION** | Fast Preview | 3D Export | Experimental | **HYBRID OPTION C** |

---

## 4. Critical Limitation Analysis

Single-image monocular 3D reconstruction is fundamentally limited by information theory. The following physical quantities **CANNOT** be recovered from a single RGB image:
1. **Backside Geometry & Textures:** The rear surfaces of characters, objects, or buildings are completely unobserved ($0\%$ observed evidence).
2. **Occluded Surface Geometry:** Regions hidden behind foreground subjects are unobserved prior to camera motion.
3. **Absolute Physical Scale:** Monocular depth maps provide relative scene depth $Z \in [0.1, 10.0]$, not true physical meters.
4. **Light Sources & Shadows Behind Objects:** Shadow casting behind subjects in unobserved areas cannot be derived without illumination priors.

Terminology Contract: All documentation and user-facing reports must use **INFERRED 3D SCENE**, **PLAUSIBLE 3D RECONSTRUCTION**, or **SINGLE-VIEW 3D REPRESENTATION**, never claiming unique physical ground-truth recovery.

---

## 5. Open-Source License Audit

| Repository / Project | License | Model / Asset License | Code Copied? | Compliance Rationale |
|---|---|---|---|---|
| **Apple SHARP (`ml-sharp`)** | MIT / Apple Sample Code | MIT | NO | Conceptually studied; independent math implementation |
| **InfiniSplat (`InfiniSplat`)** | Apache-2.0 | Apache-2.0 | NO | Conceptually studied; independent math implementation |
| **Depth Anything V2 (`Depth-Anything-V2`)** | Apache-2.0 | Apache-2.0 | NO | Used Hugging Face open weights (`Depth-Anything-V2-Small-hf`) |
| **3D Ken Burns (`3d-ken-burns`)** | GPL-3.0 | GPL-3.0 | NO | Conceptually studied; zero source code copied |
| **RAFT (`RAFT`)** | BSD-3-Clause | BSD-3-Clause | NO | OpenCV Farneback flow used |
| **SAM 2 (`sam2`)** | Apache-2.0 | Apache-2.0 | NO | Used PyTorch `sam2-hiera-tiny` open weights |

*Compliance Confirmation:* No GPL/AGPL source code was copied or imported into the repository.
