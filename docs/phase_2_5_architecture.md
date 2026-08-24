# Phase 2.5 — Architecture & Coordinate-Space Pipeline Contract

## 1. Executive Summary
This document establishes the centralized `RenderSpace` geometry contract, canonical coordinate system conventions, focal length scaling, and transform directions across all 12 stages of the 2.5D rendering pipeline.

---

## 2. Canonical Coordinate System Contract

```
                     +Y (Down)
                      |
                      |
                      |
 -X (Left) -----------+----------- +X (Right)
                      |
                      |
                      |
                     -Y (Up)

                  +Z (Forward along optical axis)
```

* **Coordinate System:** Right-handed 3D camera coordinate frame.
  - $+X$ points Right
  - $+Y$ points Down
  - $+Z$ points Forward (depth along optical axis from camera center)
* **Pixel Center Convention:** Integer pixel coordinates $(u, v)$ represent pixel centers in $[0, W-1] \times [0, H-1]$.
* **Canonical Render Resolution:** All camera intrinsics, depth representations, motion trajectories, and forward splatting operate in canonical render resolution $(W_{\text{canonical}}, H_{\text{canonical}}) = (1536, 1024)$ or input resolution $(W, H)$ with focal length $f_x = f_y = \max(W, H)$ and principal point $(c_x, c_y) = (W/2, H/2)$.
* **Depth Units:** Rendering coordinate depth $Z \in [0.1, 10.0]$ normalized scene depth, where $Z = 0.5 - 1.5$ is near foreground, $Z = 2.0 - 4.0$ is primary subject, and $Z = 5.0 - 10.0$ is background.

---

## 3. Pipeline Stage Contract Table

| Stage # | Pipeline Stage | Input Geometry | Output Geometry | Transform Direction | Units / Scale |
|---|---|---|---|---|---|
| **1** | **Depth Estimation** | RGB Image $(H, W, 3)$ | Monocular `RAW_DEPTH` | Input $\to$ Depth Space | Relative Disparity $d \in [0, 1]$ |
| **2** | **Depth Regularization** | `RAW_DEPTH` $(H, W)$ | `RENDERING_DEPTH` $(H, W)$ | Joint Bilateral Filter | Normalized $Z \in [0.1, 10.0]$ |
| **3** | **Subject Selection** | RGB $(H, W, 3)$, Depth $(H, W)$ | Refined Subject Mask $(H, W)$ | Pixel Grid $(H, W)$ | Boolean $[0, 1]$ |
| **4** | **Spatial Intelligence** | Mask $(H, W)$, Depth $(H, W)$ | `ParallaxRegion` & `MotionCouplingGroup` | Region Bounding Box | Pixel Coordinates |
| **5** | **Trajectory Planner** | Style, Strength, $(H, W)$ | Translations $T(t)$, Rotations $R(t)$ | World $\to$ Camera $SE(3)$ | Camera Units ($T_x, T_y, T_z$) |
| **6** | **Camera Intrinsics** | Image Dimensions $(H, W)$ | Intrinsics $(f_x, f_y, c_x, c_y)$ | Pinhole Parameters | Pixels ($f_x = \max(W, H)$) |
| **7** | **Backprojection** | Grid $(u, v)$, Depth $Z$ | 3D Point Cloud $P = [X, Y, Z]^T$ | 2D $\to$ 3D Camera Space | Camera Space Units |
| **8** | **SE(3) Transform** | 3D Point $P$, Pose $(R, t)$ | Transformed 3D Point $P' = R P + t$ | Camera $t_0 \to$ Camera $t_k$ | Camera Space Units |
| **9** | **Perspective Projection** | Transformed 3D Point $P'$ | Target Grid $(u', v')$ | 3D $\to$ 2D Image Space | Pixel Coordinates $(u', v')$ |
| **10** | **Subpixel Splatting** | Target $(u', v')$, Color $C$ | Accumulators `accum_col`, `accum_w` | Bilinear Weighting | Bilinear Weights $\sum w = 1.0$ |
| **11** | **Z-Buffer Fusion** | Depth $Z'$, Accumulators | Synthesized RGB Frame | Z-Ownership Reset | Normalized Color $C = \text{col}/\text{w}$ |
| **12** | **Raster Diagnostics** | Keyframe Pair $F_0 \to F_k$ | Optical Flow $d_{\text{observed}}$ | Image Pixel Flow | Isotropic Pixels |

---

## 4. Resolution Scaling & Isotropic Flow Invariance
To eliminate projected vs. raster trajectory mismatches:
$$\text{Projected Shift (px)} = f_x \cdot \left( \frac{X + T_x}{Z + T_z} - \frac{X}{Z} \right)$$
When focal length $f_x = \max(W, H)$ scales with resolution, optical flow magnitude scales proportionally:
$$\frac{d_{\text{raster}}(W_1)}{W_1} \equiv \frac{d_{\text{raster}}(W_2)}{W_2}$$
This guarantees that resolution-normalized displacement $\frac{\text{displacement\_px}}{\max(W, H)}$ remains strictly invariant across all resolutions ($1024 \times 683, 1536 \times 1024, 1920 \times 1080$).
