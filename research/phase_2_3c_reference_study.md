# PHASE 2.3C TECHNICAL REFERENCE RESEARCH STUDY

## Executive Summary
This study evaluates six key research repositories to extract conceptual algorithmic principles for 2.5D view synthesis, motion parallax, disocclusion inpainting, and optical flow measurement. In accordance with strict engineering constraints, **no code from these repositories is copied or imported**. The analysis focuses on conceptual principles and independent reproduction strategies.

---

## Reference 1: sniklaus/3d-ken-burns
- **Repository**: [https://github.com/sniklaus/3d-ken-burns](https://github.com/sniklaus/3d-ken-burns)
- **1. What it solves**: Synthesizes 3D Ken Burns effect (camera translation with depth parallax) from a single RGB image.
- **2. Relevant algorithmic idea**: Estimates monocular depth, fits a virtual 3D camera trajectory, back-projects source image points to 3D point cloud, and applies forward splatting with context-aware color inpainting.
- **3. Why relevant to our problem**: Demonstrates automatic virtual camera path planning and depth-driven perspective camera reprojection.
- **4. What we can learn**: Camera motion amplitude must scale proportionally with depth range to produce visible parallax.
- **5. What does not fit our architecture**: PyTorch-based neural inpainting pipeline that requires manual interactive UI point selection.
- **6. What must NOT be copied**: Neural network weight files, Custom CUDA C++ extension modules, GPL code.
- **7. Independent reproduction strategy**: Use continuous monocular depth with pinhole 3D back-projection $[(u-c_x)Z/f_x, (v-c_y)Z/f_y, Z]$ and Z-buffered forward subpixel splatting.
- **8. Subsystem influenced**: `v0_pipeline.py` (camera trajectory calibration & 3D back-projection).
- **9. Expected benefit**: Visually convincing camera travel with correct perspective scale growth.
- **10. Expected complexity/cost**: Low (built into existing first-principles camera model).

---

## Reference 2: pierlj/ken-burns-effect
- **Repository**: [https://github.com/pierlj/ken-burns-effect](https://github.com/pierlj/ken-burns-effect)
- **1. What it solves**: Combines depth estimation, inpainting, and dolly-zoom camera synthesis for automated video clip generation.
- **2. Relevant algorithmic idea**: Isolates foreground and background depth layers, dilates occlusion boundaries, and inpaints hidden regions before rendering novel views.
- **3. Why relevant to our problem**: Addresses disocclusion artifacts when camera travel exposes previously occluded background pixels.
- **4. What we can learn**: Creating a clean background plate *before* camera trajectory rendering prevents temporal flickering and progressive warping.
- **5. What does not fit our architecture**: Heavy dependency on external generative AI inpainting frameworks.
- **6. What must NOT be copied**: Source code, weights, or pipeline wrappers.
- **7. Independent reproduction strategy**: Conservative mask dilation combined with OpenCV Navier-Stokes/Telea RGB inpainting and Telea depth completion.
- **8. Subsystem influenced**: `spatial_intelligence/reconstruction.py` & `v0_pipeline.py` (background plate construction).
- **9. Expected benefit**: Clean background reconstruction behind moving primary subjects without black holes or halos.
- **10. Expected complexity/cost**: Low (already integrated into Phase C pipeline).

---

## Reference 3: vt-vl-lab/3d-photo-inpainting
- **Repository**: [https://github.com/vt-vl-lab/3d-photo-inpainting](https://github.com/vt-vl-lab/3d-photo-inpainting)
- **1. What it solves**: Generates 3D photography videos with wide camera motion (zoom, swing, circle, dolly-zoom) using Layered Depth Images (LDI).
- **2. Relevant algorithmic idea**: Explicit pixel connectivity handling and depth-aware context-color edge extrapolation to fill disoccluded regions.
- **3. Why relevant to our problem**: Standard point cloud forward splatting leaves disocclusion holes when the camera translates laterally or pushes forward.
- **4. What we can learn**: Edge-aware depth feathering and pixel connectivity protection suppress rubber-sheet stretching across sharp depth discontinuities.
- **5. What does not fit our architecture**: Complex multi-layer LDI representation requiring multi-stage neural network inference.
- **6. What must NOT be copied**: LDI data structures, pre-trained depth-edge network weights, license-restricted code.
- **7. Independent reproduction strategy**: Sobel depth-gradient discontinuity detection (`d_grad_mag > threshold`) to suppress forward splatting across large depth jumps, combined with pre-inpainted background plates.
- **8. Subsystem influenced**: `v0_pipeline.py` (`render_single_frame_forward_splatting`).
- **9. Expected benefit**: Elimination of tearing and rubber-sheet stretching on silhouette boundaries.
- **10. Expected complexity/cost**: Medium.

---

## Reference 4: yxuhan/AdaMPI
- **Repository**: [https://github.com/yxuhan/adampi](https://github.com/yxuhan/adampi)
- **1. What it solves**: Adaptive Multiplane Image (MPI) representation for single-image novel view synthesis.
- **2. Relevant algorithmic idea**: Discretizes scene depth into adaptive planes and blends multi-plane renderings for view synthesis.
- **3. Why relevant to our problem**: Conceptual understanding of depth discretization and plane blending.
- **4. What we can learn**: Depth quantiles ($q_{20}, q_{70}$) can isolate foreground, midground, and background raster layers for independent motion evaluation.
- **5. What does not fit our architecture**: Discretizing continuous depth into 32+ flat planes (which causes cardboard panel appearance).
- **6. What must NOT be copied**: Code or weights (AdaMPI carries non-commercial research license restrictions).
- **7. Independent reproduction strategy**: Continuous 2.5D depth field representation, using quantile depth splitting *only* for raster mask extraction and motion scoring.
- **8. Subsystem influenced**: `v0_pipeline.py` (`compute_perceptual_motion_score`).
- **9. Expected benefit**: Robust independent raster layer motion measurements without flat cardboard rendering.
- **10. Expected complexity/cost**: Low.

---

## Reference 5: princeton-vl/RAFT
- **Repository**: [https://github.com/princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)
- **1. What it solves**: High-accuracy optical flow estimation using deep recurrent all-pairs field transforms.
- **2. Relevant algorithmic idea**: Dense 2D pixel displacement vectors between frame pairs with high spatial precision.
- **3. Why relevant to our problem**: Evaluates whether rendered frames contain true structural motion vs resampling noise.
- **4. What we can learn**: Optical flow magnitude percentiles ($p_{50}, p_{90}$) provide robust displacement metrics that are immune to signed vector cancellation.
- **5. What does not fit our architecture**: Adding a heavy deep neural optical flow model as a core dependency for local rendering execution.
- **6. What must NOT be copied**: RAFT weights or model code.
- **7. Independent reproduction strategy**: Classical OpenCV Farneback dense optical flow (`cv2.calcOpticalFlowFarneback`), evaluating flow magnitude percentiles and directional variance across layer masks.
- **8. Subsystem influenced**: `v0_pipeline.py` (`compute_perceptual_motion_score`).
- **9. Expected benefit**: Accurate raster displacement measurement without vector cancellation.
- **10. Expected complexity/cost**: Low (uses built-in OpenCV).

---

## Reference 6: princeton-vl/SEA-RAFT
- **Repository**: [https://github.com/princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)
- **1. What it solves**: Ultra-fast and memory-efficient optical flow estimation.
- **2. Relevant algorithmic idea**: Efficient flow estimation with explicit uncertainty maps.
- **3. Why relevant to our problem**: Provides insights into temporal motion stability and flow uncertainty.
- **4. What we can learn**: Flow direction variance ($\sigma_{\theta}^2$) across frame sequences measures temporal motion coherence.
- **5. What does not fit our architecture**: External GPU deep learning dependency.
- **6. What must NOT be copied**: Model code or weights.
- **7. Independent reproduction strategy**: Temporal frame-to-frame flow direction tracking using Farneback flow angle histograms.
- **8. Subsystem influenced**: `v0_pipeline.py` (`analyze_image_space_motion_and_subject_fidelity`).
- **9. Expected benefit**: Detection of temporal jitter, dead zones, and frame-to-frame direction reversals.
- **10. Expected complexity/cost**: Low.

---

## Technical Comparison Matrix

| Repository | Primary Mechanism | Relevance to V0 Renderer | Integration Strategy | License Constraint |
| :--- | :--- | :--- | :--- | :--- |
| **3D Ken Burns** | Virtual camera scan + 3D point cloud | High (3D camera reprojection) | Conceptual math only | Reference only |
| **Ken Burns Effect** | Background inpainting + Dolly zoom | High (Background plate) | OpenCV Telea/Navier-Stokes | Reference only |
| **3D Photo Inpainting** | LDI + depth-aware edge extrapolation | High (Disocclusion handling) | Sobel depth-gradient thresholds | Reference only |
| **AdaMPI** | Adaptive Multiplane Images | Medium (Layer quantile masks) | Depth quantile mask splitting | Non-commercial (Reference only) |
| **RAFT** | Recurrent optical flow | High (Raster motion analysis) | OpenCV Farneback magnitude percentiles | Reference only |
| **SEA-RAFT** | Fast flow + uncertainty estimation | Medium (Temporal coherence) | OpenCV flow angle variance | Reference only |
