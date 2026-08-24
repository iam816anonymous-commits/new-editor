# Phase 2.4 Baseline Architecture Audit Report — Semantic + Geometric Parallax Intelligence

**Date:** August 15, 2026
**Status:** COMPLETE — BASELINE ARCHITECTURE AUDIT
**Author:** First-Principles Cinematic 2.5D Parallax Renderer (V0) Pipeline

---

## Executive Summary
This audit inspects the existing spatial intelligence, subject selection, scene graph, and motion mapping subsystems in `spatial_intelligence/` and `v0_pipeline.py`. It provides code-level answers to the 10 core architectural questions, identifying how the renderer currently transitions from semantic masks to 3D motion, and pinpointing where independent cutout sliding can occur.

---

## Structural Architecture Questions & Answers

### 1. What is currently considered an entity?
- **Definition:** An `Entity` (`spatial_intelligence/schemas.py`) represents a consolidated spatial mask region extracted from SAM 2 candidate masks.
- **Attributes:** `entity_id`, `name`, `entity_class` (`RENDERABLE_ENTITY` or `ANALYSIS_REGION`), `semantic_role` (`PRIMARY_SUBJECT`, `SECONDARY_SUBJECT`, `BACKGROUND_ENVIRONMENT`, `SURFACE_STRUCTURE`), `layer_role`, `mask`, `area`, `bounding_box`, `depth_mean`, `depth_median`, `depth_std`, `trust_score`.

### 2. What is currently considered a layer?
- **Definition:** A `LayerRole` (`spatial_intelligence/schemas.py`) is a coarse spatial depth partition: `PRIMARY_SUBJECT`, `FOREGROUND`, `MIDGROUND`, `BACKGROUND`.
- **Assignment:** Assigned in `v0_pipeline.py` using relative monocular depth quantiles ($q_{20}, q_{70}$) across background regions.

### 3. Which entities have one mean depth versus a spatial depth field?
- **Continuous Depth Fields:** The renderer maintains continuous 2D depth fields $Z(u, v)$ for all pixels via Joint Bilateral Filtering (`edge_aware_depth_refinement`).
- **Entity Summaries:** Scene graph entities store summary statistics (`depth_mean`, `depth_median`, `depth_std`), but 3D reprojection in forward splatting evaluates pixel-level continuous depth $Z(u, v)$ rather than flat entity planes.

### 4. Where are depth boundaries represented?
- **Representation:** Fused edge maps ($\nabla \text{depth} + \text{Canny RGB edges}$) are computed in `spatial_intelligence/edge_detector.py` and boundary risk maps in `compute_boundary_risk_map()`.

### 5. Where are occlusion boundaries represented?
- **Representation:** Pairwise occlusion relationships (`FRONT_OF`, `OCCLUDES`, `SUPPORTS`) are inferred in `spatial_intelligence/relationship_inferencer.py` using multi-signal intersection and depth ordering ($Z_A < Z_B$).

### 6. How are compound subjects represented?
- **Representation:** Candidate masks that share spatial containment, IoU overlap, or support contact are merged in `subject_selection/candidate_grouper.py` into a single compound subject mask.

### 7. How are object parts represented?
- **Representation:** `EntityPart` (`spatial_intelligence/schemas.py`) tracks sub-mask candidates (e.g. face, hand, ornaments) belonging to a parent entity, storing `part_id`, `parent_entity_id`, `mask`, and `relative_depth`.

### 8. How is motion currently assigned to entities?
- **Assignment:** Motion is assigned via `construct_layer_motion_map()`, which paints scalar motion multipliers across pixel grids based on layer role (`PRIMARY_SUBJECT` = 0.35x, `FOREGROUND` = 2.20x, `MIDGROUND` = 1.50x, `BACKGROUND` = 1.00x).

### 9. Where is semantic information converted into motion?
- **Conversion:** In `construct_layer_motion_map()`, the semantic subject mask locks pixels to the `PRIMARY_SUBJECT` motion multiplier (0.35x), while remaining non-subject pixels receive depth-quantile layer multipliers before $SE(3)$ camera pose reprojection.

### 10. Where can a region accidentally become an independent moving cutout?
- **Vulnerability:** When adjacent semantic entities or candidate parts receive different layer motion multipliers without explicit attachment coupling, boundary shear or "2D layer sliding" can occur.
- **Phase 2.4 Solution:** Introduce `ParallaxRegion`, `AttachmentType`, and `MotionCouplingGroup` to ensure connected parts share a unified 3D motion field governed by exact pixel-level continuous 3D depth geometry.
