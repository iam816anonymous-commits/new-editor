# Spatial Intelligence Subsystem Architecture (Phase 1)
First-Principles Cinematic 2.5D Parallax Renderer (V0)

## 1. Executive Summary & Repository Audit

The V0 pipeline implements a deterministic 2.5D view synthesis renderer.

### Ingestion & Processing Pipeline
1. **CLI & Image Ingestion (`v0_pipeline.py`)**: Accepts user-provided input image, validates image decoding, calculates SHA-256 hash (`output/<hash>/`), and detects system FFmpeg binary.
2. **Monocular Depth Estimation (`v0_pipeline.py`)**: Executes real `Depth Anything V2 Small` inference (`depth-anything/Depth-Anything-V2-Small-hf`), percentile clips outliers (1st/99th), normalizes to rendering coordinate depth $Z \in [0.1, 10.0]$, performs Joint Bilateral edge-guided depth refinement, and calculates a depth gradient confidence map.
3. **Semantic Subject Selection (`subject_selection/` package)**:
   - Generates candidate masks using SAM 2 Hiera-Tiny (`facebook/sam2-hiera-tiny`) with 5x5 grid point prompts, foreground depth saliency points, and multi-scale box prompts.
   - Extracts candidate features (`mask_area`, `bbox`, `norm_centroid`, `border_touch_ratio`, `depth_saliency`, `background_contamination_score`, `fragmentation_score`).
   - Evaluates multi-signal scores combining SAM confidence, centrality, depth saliency, scale coverage, relative visual prominence, and compound support minus border, contamination, and environmental isolation penalties.
   - Groups compatible compound candidates (e.g. Vishnu + Shesha) and performs edge-preserving mask refinement.
   - Evaluates a mask acceptance gate (`MaskValidationResult`), halting pipeline execution before Phase C background reconstruction if candidate selection is ambiguous or invalid.
4. **Background Plate Reconstruction & Provenance (`v0_pipeline.py`)**:
   - Dilates subject mask conservatively with RGB edge guidance (`dilated_mask`).
   - Inpaints RGB background plate (`background_plate.png`) and background depth (`background_depth.png`).
   - Tracks pixel provenance (`1.0` = OBSERVED, `0.0` = RECONSTRUCTED) and boundary risk (`boundary_risk_map.png`).
5. **Closed-Loop Trajectory Planning & Rendering (`v0_pipeline.py`)**:
   - Derives camera intrinsics ($f_x = f_y = \max(W, H)$, $c_x = W/2, c_y = H/2$).
   - Generates $C^1$-continuous smooth trajectories ($P(0)=P(1), V(0)=V(1)=0$).
   - Enforces closed-loop disparity ceiling safety envelopes (`Subtle` 1.5%, `Cinematic` 3.0%, `Strong` 5.0% width disparity).
   - Forward subpixel splatting with bilinear weight distribution and deterministic Z-buffering ($Z_{\text{closer}}$ wins).
   - Encodes 48-frame 24 FPS MP4 video using FFmpeg and validates video metadata.

---

## 2. Identified Limitations of Baseline Parallax Output

While the baseline output exhibits strong temporal stability and rigid subject preservation, the rendered parallax is limited due to:

1. **Continuous Surface Monocular Depth**: The monocular depth map is mapped continuously ($Z \in [0.1, 10.0]$) without explicit 2.5D layer/entity decomposition or part-level depth field structure.
2. **Uniform Camera-Space Reprojection**: Forward splatting back-projects all pixels using one unstratified depth map.
3. **Conservative Disparity Envelope Scaling**: Closed-loop motion planning scales trajectory magnitudes down to guarantee per-frame disparity ceilings. Because depth $Z$ lacks entity-level layer separation, differential disparity between foreground and background is compressed ($\sim 10\text{--}20\text{px}$ screen displacement).

---

## 3. Proposed Spatial Intelligence Layer Architecture

The `spatial_intelligence/` subsystem introduces explicit scene understanding without modifying the frozen `subject_selection` package or introducing fake heuristic fallbacks.

```
                    IMAGE & DEPTH MAP
                           │
                           ▼
                  SUBJECT SELECTION (FROZEN)
                           │
                           ▼
                 SPATIAL INTELLIGENCE
            ┌─────────────────────────────┐
            │ 1. Entity & Part Extractor  │
            │ 2. Scene Graph Construction │
            │ 3. Spatial Inference        │
            │ 4. Structured Depth Field   │
            │ 5. Occlusion Model          │
            │ 6. Pinhole Camera Model     │
            │ 7. Temporal Spatial Tracker │
            └──────────────┬──────────────┘
                           │
                           ▼
            BACKGROUND RECONSTRUCTION & RENDER
```

### Key Subsystem Contracts (`spatial_intelligence/schemas.py`)
- `Entity`: A semantically meaningful scene component (`id`, `name`, `mask`, `bbox`, `centroid`, `depth_stats`, `is_primary_subject`).
- `EntityPart`: Sub-components of an entity (`id`, `entity_id`, `name`, `mask`, `depth_stats`).
- `SpatialRelationship`: Directed edge between entities/parts (`subject_id`, `target_id`, `relation_type`, `confidence`, `evidence`).
- `SceneGraph`: Graph representation supporting `IN_FRONT_OF`, `BEHIND`, `SUPPORTS`, `ATTACHED_TO`, `PART_OF`, `OVERLAPS`, `OCCLUDES`, `OCCLUDED_BY`, `NEAR`, `FAR_FROM`, `SAME_COMPOUND_SUBJECT`.
- `DepthField`: 2.5D spatial depth representation holding background depth, subject depth, internal subject depth variation, foreground depth, and uncertainty map.
- `OcclusionRelationship`: Represents occluder, occluded entity, boundary zone, and disocclusion exposure risk.
- `CameraModel`: Pinhole perspective projection model holding focal length, camera position, orientation, and 3D-to-2D projection.
- `SpatialConfidence`: Multi-layered spatial confidence (relationship confidence, depth field confidence, occlusion confidence, overall spatial confidence).
- `SpatialDiagnostics`: Exportable structured JSON report and diagnostic maps (`spatial_scene.json`, `spatial_relationships.json`, `depth_field.png`, `depth_uncertainty.png`, `occlusion_map.png`, `camera_path.json`, `spatial_diagnostics.json`).

---

## 4. Testing & Migration Strategy

1. **Frozen Baseline Preservation**: All 41 existing pytest unit and integration tests must pass continuously without modification to `subject_selection/` core logic or parameters.
2. **Deterministic Parallax Acceptance Test**: Add a synthetic controlled camera motion test asserting $Displacement_{\text{fg}} > Displacement_{\text{sub}} > Displacement_{\text{bg}}$ under perspective projection.
3. **Ablation Testing**:
   - Collapse depth field to single plane $\rightarrow$ verify parallax degradation.
   - Disable occlusion model $\rightarrow$ verify exposure risk degradation.
   - Disable camera model $\rightarrow$ verify projection failure.
4. **Fail-Safe Gate**: If spatial confidence is below threshold, return a degraded but valid 2.5D representation with explicit uncertainty bounds instead of hallucinating fake geometry.
