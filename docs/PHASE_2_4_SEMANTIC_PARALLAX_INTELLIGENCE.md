# Phase 2.4 — Semantic + Geometric Parallax Intelligence Documentation

**Date:** August 15, 2026
**Status:** COMPLETE — PHASE 2.4 DOCUMENTATION
**Author:** First-Principles Cinematic 2.5D Parallax Renderer (V0) Pipeline

---

## 1. Executive Summary
Phase 2.4 upgrades the renderer from coarse per-layer motion scaling to **Semantic + Geometric Parallax Intelligence**. Rather than treating semantic objects as flat 2D cutouts that slide independently across discrete planes, the system constructs a spatially and semantically coherent 2.5D scene representation. Connected parts of rigid objects share unified 3D motion coupling groups, while individual pixels obey their exact continuous 3D depth geometry under $SE(3)$ camera transformations.

---

## 2. Core Architectural Components

### A. Parallax Region Representation (`spatial_intelligence/parallax_region.py`)
Replaces flat entity planes with `ParallaxRegion`, capturing continuous spatial depth statistics:
- **Depth Distributions:** `depth_mean`, `depth_median`, `depth_p10`, `depth_p25`, `depth_p50`, `depth_p75`, `depth_p90`, `depth_std`, `depth_iqr`.
- **Depth Structure Types:** `UNIFORM_DEPTH`, `GRADIENT_SURFACE`, `MULTI_DEPTH_OBJECT`, `DEPTH_DISCONTINUITY`, `UNCERTAIN_DEPTH`.
- **Boundary Attributes:** `local_depth_gradient` ($|\nabla \text{depth}|$), `depth_discontinuity_score`, `occlusion_boundary_mask`, `disocclusion_risk`, `rigidity_score`, `attachment_score`, `support_score`.

### B. Attachment Graph & Motion Coupling (`spatial_intelligence/parallax_coupling.py`)
Infers physical connections between scene parts to eliminate "cardboard layer sliding":
- **Attachment Types:** `RIGIDLY_ATTACHED`, `SOFTLY_ATTACHED`, `INDEPENDENT`, `SUPPORTED_BY`, `RESTING_ON`, `BEHIND`, `IN_FRONT_OF`, `PART_OF`, `SAME_SURFACE`, `UNKNOWN`.
- **Motion Coupling Groups:** `MotionCouplingGroup` links semantically connected parts (e.g. face, torso, hands, ornaments) into a shared camera-induced 3D motion field.
- **Motion Eligibility:** `PRIMARY_CAMERA_PARALLAX`, `SECONDARY_PARALLAX`, `BACKGROUND_PARALLAX`, `STATIC_REFERENCE`, `ATTACHED_TO_GROUP`, `INDEPENDENT_ELEMENT`, `MOTION_SUPPRESSED`, `UNCERTAIN`.

### C. Occlusion Boundaries & Disocclusion Forecasting
- **Occlusion Boundaries:** `OcclusionBoundary` tracks foreground/background region pairs, depth delta, confidence, and expected disocclusion exposure width.
- **Trajectory-Aware Forecasting:** `forecast_disocclusion_regions()` predicts newly exposed pixel area for candidate camera translations before rendering.

---

## 3. Motion-Eligibility Table (Example: Vishnu/Space Scene)

| Region ID | Semantic Role | Depth Mean | Depth Std | Depth Structure | Attachment Type | Motion Eligibility | Motion Coupling Group | Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `REG_00_PRIMARY_SUBJECT` | `PRIMARY_SUBJECT` | $0.571$ | $0.502$ | `MULTI_DEPTH_OBJECT` | `RIGIDLY_ATTACHED` | `PRIMARY_CAMERA_PARALLAX` | `GROUP_PRIMARY_SUBJECT` | $0.950$ |
| `REG_01_COSMIC_BACKGROUND` | `BACKGROUND_ENVIRONMENT` | $7.950$ | $1.210$ | `GRADIENT_SURFACE` | `INDEPENDENT` | `BACKGROUND_PARALLAX` | `GROUP_ENV_01` | $0.850$ |

---

## 4. Fake Parallax Detection
The visual quality validation engine (`spatial_intelligence/visual_quality.py`) evaluates explicit fake parallax failure codes:
- `FAKE_PARALLAX`: Flagged when semantically connected parts shear apart without depth justification.
- `MOTION_ATTACHMENT_VIOLATION`: Flagged when attached regions move in opposing directions.
- `DEPTH_MOTION_INCONSISTENCY`: Flagged when a distant region moves more than a closer foreground region without geometric explanation.
- `UNJUSTIFIED_LAYER_TRANSLATION`: Flagged when a flat 2D layer translation is applied without 3D perspective division.

---

## 5. Diagnostic Artifact Package (`output/<hash>/spatial_analysis/`)

Exports 7 JSON reports and 11 visual diagnostic PNG plots per render:
- `scene_analysis.json`
- `parallax_regions.json`
- `attachment_graph.json`
- `motion_coupling.json`
- `motion_eligibility.json`
- `disocclusion_forecast.json`
- `00_primary_spatial_overlay.png` (Primary Spatial Overlay showing subject, rigid groups, depth regions, occlusion boundaries, and motion vectors)
- `01_depth_raw.png`
- `02_depth_edges.png`

---

## 6. Known Limitations
- **Fine Hair / Filigree Boundaries:** Extremely thin semi-transparent structures (e.g. single-pixel filigree) rely on SAM 2 mask boundary precision and bilateral depth filter sharpness.
