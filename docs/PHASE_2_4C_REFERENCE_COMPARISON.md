# Phase 2.4C — Reference Comparison & Architectural Decision Audit

## 1. Executive Summary & Objective
Phase 2.4C evaluates six open-source reference implementations to inform the mathematical, geometric, and semantic architecture of our First-Principles Cinematic 2.5D Parallax Renderer.

The primary objective is to transition from ungrounded layer displacement multipliers to a **Reference-Guided Surface-Coherent 3D Camera Projection Model** that eliminates texture swimming, surface shearing, edge crawling, and artificial layer translations while preserving rigid subject semantics and continuous environmental depth.

---

## 2. Reference Study Analysis

### 2.1 Reference 1: DepthFlow (`BrokenSource/DepthFlow`)
* **Repository & Source:** `https://github.com/BrokenSource/DepthFlow` (GPL-3.0 License)
* **Priority:** VERY HIGH
* **Relevant Files:** `depthflow/scene.py`, `depthflow/resources/depthflow.glsl`
* **Algorithmic Insights & Math:**
  - **Camera & Ray Intersection:** DepthFlow uses a GPU GLSL fragment shader model where rays are cast per pixel into a normalized depth displacement field $Z(u, v) \in [0, 1]$.
  - **Dolly/Zoom Coupling:** Camera movement is parameterized by normalized offsets $(dx, dy)$ and depth-height scaling $S_z$.
  - **Image-Space Sampling:** Rather than translating 2D image cards, DepthFlow shifts texture sampling coordinates $(u', v') = (u, v) + (dx, dy) \cdot Z(u, v)$, modulating displacement directly by depth height.
  - **Out-of-Bounds Handling:** Uses boundary clamping and edge mirrored padding (`GL_CLAMP_TO_EDGE` / `GL_MIRRORED_REPEAT`) to prevent edge tearing.
* **Our Independent Application:**
  - Adopted depth-driven per-pixel ray sampling concepts into our forward subpixel splatting engine (`v0_pipeline.py`).
  - Derived screen displacement $d_{\text{expected}}$ directly from 3D pinhole camera backprojection $P' = R P + t$ rather than GLSL 2D texture coordinate offset tricks.

### 2.2 Reference 2: 3D Ken Burns (`sniklaus/3d-ken-burns`)
* **Repository & Source:** `https://github.com/sniklaus/3d-ken-burns` (GPL-3.0 License)
* **Priority:** VERY HIGH
* **Relevant File:** `models/pointcloud-inpainting.py`
* **Algorithmic Insights & Pipeline Trace:**
  - **Pipeline:** Disparity $d \to$ Depth $Z = 1/d \to$ 3D Point Cloud $P = [X, Y, Z] \to$ Virtual Camera Transform $SE(3) \to$ Novel View Projection $u' = f_x X'/Z' + c_x \to$ Point Cloud Rasterization $\to$ Valid Pixel Mask $\to$ Disocclusion Detection $\to$ Contextual Inpainting.
  - **Disocclusion & Inpainting:** Identifies disoccluded regions where projected sample density falls below threshold, inpainting background depth and RGB.
* **Our Independent Application:**
  - Compared point cloud rasterization against our forward subpixel splatting with Z-buffering.
  - Adopted strict Z-ownership resets (`accum_col[v, u] = 0, accum_w[v, u] = 0`) when closer surfaces win in the Z-buffer to eliminate double ownership and ghosting.

### 2.3 Reference 3: 3D Photo Inpainting (`vt-vl-lab/3d-photo-inpainting`)
* **Repository & Source:** `https://github.com/vt-vl-lab/3d-photo-inpainting` (MIT License)
* **Priority:** VERY HIGH
* **Algorithmic Insights:**
  - **Layered Depth Image (LDI) Representation:** Represents scene as explicit connected depth meshes. Pixels with small depth gradients remain connected as continuous surfaces, while large depth jumps are split into distinct depth layers with explicit occlusion relationships.
  - **Pixel Connectivity:** Prevents rubber-sheet stretching across depth discontinuities by disconnecting mesh edges where $\Delta Z > T_{\text{discont}}$.
* **Our Independent Application:**
  - Adopted depth discontinuity edge detection ($\Delta Z > T$) to break splatting connections across sharp object boundaries while maintaining continuous depth fields inside connected regions.

### 2.4 Reference 4: Parallax Maker (`provos/parallax-maker`)
* **Repository & Source:** `https://github.com/provos/parallax-maker` (MIT License)
* **Priority:** HIGH
* **Algorithmic Insights:**
  - **Semantic Card-Based Layers:** Uses SAM segmentation masks to carve scenes into discrete 2.5D planar cards.
  - **Limitation:** Carving scenes strictly into flat 2D cards produces artificial "cardboard layer sliding" where subjects lose internal 3D geometry.
* **Our Independent Application:**
  - Replaced flat card translations with continuous 3D local depth fields ($Z(u,v)$) inside semantic regions (`PRIMARY_SUBJECT`, `FOREGROUND`, `MIDGROUND`, `BACKGROUND`), preserving internal 3D shape (e.g. head, torso, ornaments).

### 2.5 Reference 5: Improved Ken Burns (`pierlj/ken-burns-effect`)
* **Repository & Source:** `https://github.com/pierlj/ken-burns-effect` (MIT License)
* **Priority:** MEDIUM/HIGH
* **Algorithmic Insights:**
  - Smooth camera trajectory keyframing with crop window bounds calculation to avoid black border exposures.
* **Our Independent Application:**
  - Integrated quintic smoothstep trajectory easing ($s_{\text{quintic}}(t) = 6t^5 - 15t^4 + 10t^3$) and safe motion envelope bounds into `plan_safe_motion_trajectory()`.

### 2.6 Reference 6: Depth-Based Parallax Effect (`r-gheda/depth-based-parallax-effect`)
* **Repository & Source:** `https://github.com/r-gheda/depth-based-parallax-effect` (MIT License)
* **Priority:** MEDIUM
* **Algorithmic Insights:**
  - Edge-aware depth map smoothing using bilateral filtering to align depth edges with RGB object boundaries.
* **Our Independent Application:**
  - Integrated Joint Bilateral Filtering guided by RGB Canny edges in `spatial_intelligence/depth_field.py` (`RAW_DEPTH` vs `RENDERING_DEPTH`).

---

## 3. Comparative 19-Point Pipeline Matrix

| Pipeline Stage | Our Renderer (V0 Pipeline) | Reference 1 (DepthFlow) | Reference 2 (3D Ken Burns) | Reference 3 (3D Photo Inpainting) |
|---|---|---|---|---|
| **1. Camera Representation** | Pinhole ($f_x, f_y, c_x, c_y$) | Normalized Offset / GLSL | Pinhole Point Cloud | Layered Mesh Camera |
| **2. Camera Convention** | Right-handed (+X Right, +Y Down, +Z Forward) | 2D UV Displacement | 3D Camera Frame | 3D LDI Frame |
| **3. Camera Trajectory** | Quintic Smoothstep ($s_{\text{quintic}}$) + Closed-Loop Safety | Parametric Sine/Cosine | Keyframe Spline | Keyframe Spline |
| **4. Depth Representation** | Dual (`RAW_DEPTH` vs `RENDERING_DEPTH`) | Normalized Depth Texture | Disparity $d = 1/Z$ | Layered Depth Image |
| **5. Depth Normalization** | Scene-Adaptive Floor Clamp ($Z \in [0.1, 10.0]$) | $[0, 1]$ Height Map | Inverted Disparity | Depth Range $[0, 1]$ |
| **6. Depth Refinement** | Edge-Guided Joint Bilateral Filtering | Bilateral Filtering | Inpainting Mesh Filter | Contour Mesh Edge Filter |
| **7. Surface Representation** | Continuous Local Depth Field | GLSL Sampling Grid | 3D Point Cloud Mesh | Layered Mesh Surfaces |
| **8. Semantic Regions** | `PRIMARY_SUBJECT`, `FOREGROUND`, etc. | None (Global Texture) | None (Global Point Cloud) | Occlusion Context Mesh |
| **9. Attachment/Coupling** | `MotionCouplingGroup` & `AttachmentType` | None | None | Pixel Connectivity Mesh |
| **10. Perspective Projection** | $P' = R P + t, u' = f_x X'/Z' + c_x$ | 2D UV Offset Approximation | $P' = R P + t, u' = f_x X'/Z' + c_x$ | 3D Mesh Projection |
| **11. Warp/Render Method** | Forward Subpixel Splatting + Bilinear Weights | GLSL Fragment Texture Lookup | Point Cloud Splatting | Mesh Rasterization |
| **12. Visibility Reasoning** | Deterministic Z-Buffer | Z-Depth Map | Point Cloud Z-Buffer | LDI Layer Ordering |
| **13. Occlusion Reasoning** | Multi-Signal Occlusion Model | None | Point Cloud Hole Detection | LDI Context Hole Mask |
| **14. Disocclusion Gen.** | Pre-rendered Background Plate Inpainting | Mirrored Padding | Point Cloud Context Inpaint | Mesh Edge Inpainting |
| **15. Inpainting** | OpenCV Telea Inpainting | Mirrored Edge Clamp | Contextual Inpainting Model | Contextual Inpainting Model |
| **16. Border Handling** | Border Reflect Padding + Telea Fill | Mirrored Texture Repeat | Crop Window Overscan | Mesh Edge Extension |
| **17. Temporal Rendering** | Immutable Reference Scene ($F_0 \to F_n$) | GLSL Time Progression | Novel View Frame Sequence | Novel View Frame Sequence |
| **18. Motion Measure** | Peak Optical Flow Vector Tracking across Keyframes | None | Optical Flow Validation | Point Disparity Error |
| **19. Artifact Detection** | 2.5D Artifact Codes (`TEXTURE_SWIM`, etc.) | None | Point Discontinuity Mask | Discontinuity Mask |

---

## 4. Architecture Evaluation & Decision

### Evaluation of Architecture Alternatives:
* **Option A — Layer-based 2.5D renderer (Discarded):**
  - Carves images into flat 2D cards and translates them independently.
  - *Failure:* Causes severe "cardboard panel sliding" and destroys local 3D subject geometry.
* **Option B — Pure depth-aware camera renderer (Discarded):**
  - Backprojects all pixels using raw depth without semantic grouping or attachment constraints.
  - *Failure:* Monocular depth noise causes connected body parts (e.g. face vs ornaments) to shear apart or swim independently.
* **Option C — HYBRID: Semantic understanding for region relationships + Depth-aware camera projection for actual motion (SELECTED):**
  - Uses semantic graph relationships (`MotionCouplingGroup`, `AttachmentType`) to enforce motion cohesion across connected surfaces (e.g. body, garments, ornaments).
  - Uses continuous local depth fields $Z(u,v)$ for exact 3D pinhole camera backprojection ($P' = R P + t$), preserving true 3D perspective geometry and parallax.

### Final Decision:
Option C (HYBRID) is selected and implemented as our core rendering architecture.
