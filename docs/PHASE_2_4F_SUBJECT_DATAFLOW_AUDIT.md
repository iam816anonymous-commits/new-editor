# Phase 2.4F — Subject Completeness & Selection Data Flow Audit

## 1. Executive Summary
This document records a forensic audit of the subject data flow across the entire selection, refinement, spatial intelligence, and 2.5D rendering pipeline (`subject_selection/`, `spatial_intelligence/`, and `v0_pipeline.py`).

---

## 2. Complete Stage-by-Stage Data Flow Analysis

### Stage 1: SAM2 Candidate Generation (`generate_candidate_masks_sam2`)
* **Input:** RGB image array $(H, W, 3)$, Depth map $(H, W)$.
* **Output:** Deduplicated candidate masks list (`mask_bool`, `sam_score`, `prompt_origin`).
* **Coordinate System:** Pixel grid $(y, x) \in [0, H-1] \times [0, W-1]$.
* **Information Loss Risk:** Prompt grid coarseness (5x5 point grid) or box prompt bounds can miss small, isolated, or thin extremities (e.g. crown points, finger tips, thin weapons).
* **Gate / Safeguard:** Multi-prompt sampling (grid points + 5 depth saliency foreground points + 3 bounding boxes) with IoU deduplication ($> 0.85$).

### Stage 2: Candidate Feature Extraction & Relative Ranking (`extract_candidate_features`)
* **Input:** Candidate boolean mask $(H, W)$, Depth map $(H, W)$, RGB array $(H, W, 3)$.
* **Output:** Typed `CandidateFeatures` (area, centroid, depth mean/std, boundary contact, background contamination score, relative visual prominence, compound support).
* **Coordinate System:** Normalized image coordinates $[0.0, 1.0]$.
* **Information Loss Risk:** Candidates touching border edges or with background overlap might be assigned high contamination penalties if border distance is small.
* **Gate / Safeguard:** Relative batch feature normalization (`compute_batch_relative_features`) evaluating foreground depth clustering distance and compound support.

### Stage 3: Candidate Grouping & Pairwise Compatibility (`generate_candidate_groups`)
* **Input:** Candidate feature list, candidate scores list, candidate mask map.
* **Output:** `CandidateGroup` instances containing merged compound subject masks and combined scores.
* **Coordinate System:** Pixel grid $(H, W)$.
* **Information Loss Risk:** Top-$K$ candidate truncation or strict spatial proximity cutoffs could discard small attached components.
* **Gate / Safeguard:** Evaluates top 16 candidate features (`top_feats[:16]`) across candidate graph with pairwise compatibility scoring ($\text{compat} \ge 0.50$).

### Stage 4: Mask Refinement & Thin-Structure Protection (`refine_subject_mask`)
* **Input:** Merged boolean subject mask, RGB array $(H, W, 3)$.
* **Output:** Refined boolean subject mask $(H, W)$.
* **Coordinate System:** Pixel grid $(H, W)$.
* **Information Loss Risk:** Morphological closing/opening could erode thin structures if done without edge constraints.
* **Gate / Safeguard:** RGB Canny edge-guided boundary protection (`Canny(gray, 40, 120)`) preventing expansion or erosion across sharp color edges while preserving thin structures.

### Stage 5: Mask Validation Gate (`validate_selected_subject_mask`)
* **Input:** Selected CandidateGroup, candidate features map, score margin.
* **Output:** `MaskValidationResult` (`validation_status`, `confidence`, `is_valid`).
* **Information Loss Risk:** Score margin $< 0.05$ or area ratio $< 0.01$ raises `UNCERTAIN` / `REJECT`.
* **Gate / Safeguard:** Hard validation gate in `select_semantic_subject` raising explicit error if invalid before Phase C background reconstruction.

### Stage 6: Spatial Entity & ParallaxRegion Creation (`construct_parallax_regions`)
* **Input:** Spatial entities, depth map $(H, W)$, RGB array $(H, W, 3)$.
* **Output:** `ParallaxRegion` instances with local depth stats, depth gradients, and boundary masks.
* **Information Loss Risk:** High local depth gradients could misclassify smooth curved surfaces as depth discontinuities.
* **Gate / Safeguard:** Edge-alignment status classification (`classify_depth_rgb_edge_alignment`) validating depth edges against dilated RGB Canny boundaries.

### Stage 7: Motion Coupling & 3D View Synthesis (`render_single_frame_forward_splatting`)
* **Input:** Subject mask, background plate, rendering depth, camera $SE(3)$ pose.
* **Output:** Synthesized RGB frame, rendered Z-buffer, output provenance map.
* **Information Loss Risk:** Disocclusion holes at subject boundaries.
* **Gate / Safeguard:** Deterministic Z-buffer ownership reset (`accum_col[v, u] = 0, accum_w[v, u] = 0`) and Telea inpainting on moving frames.
