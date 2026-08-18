# Spatial Intelligence Subsystem Architecture (Phase 1.5 Hardening)
First-Principles Cinematic 2.5D Parallax Renderer (V0)

## 1. Executive Summary & Hardening Overview

In Phase 1.5, the Spatial Intelligence Layer was hardened to solve candidate proposal sprawl and graph density explosion.

### Problem Addressed
- **Candidate Sprawl**: SAM 2 proposals were previously treated 1:1 as scene entities, generating 50+ raw candidates for simple scenes.
- **Graph Density Explosion**: $53$ entities generated $\approx 5,649$ relationships because reciprocal pairs (`IN_FRONT_OF` + `BEHIND`, `OCCLUDES` + `OCCLUDED_BY`, `OVERLAPS`, `NEAR`) were stored as independent edges for all entity pairs without canonical deduplication or trust filtering.

### Solution Architecture
We introduced explicit distinction between raw proposals, consolidated entities, trusted entities, and canonical sparse relationships while preserving **CONTINUOUS DEPTH** as the primary geometric representation.

```
                   RGB & CONTINUOUS DEPTH
                             │
                             ▼
               Segmentation Candidate Proposals
                             │
                             ▼
                Candidate Consolidation Stage
            (IoU / Containment / Depth Filtering)
                             │
                             ▼
                  Trusted Spatial Entities
               (Semantic Roles & Trust Scores)
                             │
                             ▼
                Sparse Canonical Scene Graph
            (Multi-Signal Occlusion & Deduplication)
                             │
                             ▼
              2.5D View Synthesis (Forward Splatting)
```

---

## 2. Core Concepts & Data Contracts (`spatial_intelligence/schemas.py`)

1. **`SegmentationCandidate`**: Raw, unconsolidated candidate proposal from SAM 2 or prompt grid.
2. **`SpatialEntity`**: Consolidated spatial region formed by merging duplicate or contained segmentation proposals.
3. **`EntityTrustScore`**: Multi-signal trust breakdown separate from raw SAM segmentation confidence.
4. **`SemanticRole`**: Coarse semantic classification (`PRIMARY_SUBJECT`, `FOREGROUND_OBJECT`, `SECONDARY_OBJECT`, `MIDGROUND_OBJECT`, `BACKGROUND_REGION`, `EMPTY_BACKGROUND`).
5. **`RenderRelevance`**: Importance classification for view synthesis (`CRITICAL`, `USEFUL`, `LOW`, `IGNORE`).
6. **`SpatialRelationship`**: Canonical directed edge with supporting metrics and evidence.

---

## 3. Candidate Consolidation Algorithm (`spatial_intelligence/entity_consolidator.py`)

Raw proposals are consolidated into a small set ($N \approx 5\text{--}15$) of trusted entities:

1. **Primary Subject Protection**: Primary subject mask is preserved as Entity 1 (`PRIMARY_SUBJECT`). Proposals overlapping $>60\%$ with the primary subject are absorbed into its source candidate list without fragmenting primary geometry.
2. **Containment & Duplicate Merging**: Non-primary proposals with $\text{IoU} \ge 0.70$ or mask containment $\ge 0.85$ and compatible depth ($|\Delta Z| < 1.0$) are merged into single spatial entities.
3. **Environmental Noise Rejection**: Tiny fragments ($<0.5\%$ image area) and massive environmental masks ($>85\%$ image area) lacking depth boundaries are rejected.

---

## 4. Entity Trust Scoring Formula (`spatial_intelligence/entity_trust.py`)

Entity trust score is calculated deterministically:

$$\text{entity\_trust\_score} = 0.25 \cdot Q_{\text{mask}} + 0.25 \cdot C_{\text{depth}} + 0.20 \cdot C_{\text{boundary}} + 0.15 \cdot U_{\text{candidate}} + 0.15 \cdot R_{\text{render}}$$

- **$Q_{\text{mask}}$ (Mask Quality)**: Bounding box compactness and preferred area coverage ($1\%\text{--}60\%$).
- **$C_{\text{depth}}$ (Depth Coherence)**: Standard deviation relative to depth span and surrounding depth ring separation.
- **$C_{\text{boundary}}$ (Boundary Coherence)**: Alignment with Canny color edges and depth confidence values along the boundary silhouette.
- **$U_{\text{candidate}}$ (Uniqueness)**: Candidate proposal provenance.
- **$R_{\text{render}}$ (Render Relevance)**: Centrality and proximity to foreground depth $Z$.

---

## 5. Multi-Signal Occlusion Gate & Canonical Sparse Graph (`spatial_intelligence/relationship_inferencer.py`)

### Sparse Canonical Graph Rules
1. **Canonical Edge Ordering**: Relationships are deduplicated by pair ordering $(\min(\text{id}_1, \text{id}_2), \max(\text{id}_1, \text{id}_2))$. Reciprocal duplicates (`BEHIND` for `FRONT_OF`, `OCCLUDED_BY` for `OCCLUDES`) are represented as single canonical relationships.
2. **`OVERLAPS` Disambiguation**: `OVERLAPS` records 2D mask overlap but does **NOT** automatically imply `OCCLUDES`.
3. **Multi-Signal Occlusion Gate**: `A OCCLUDES B` requires passing a multi-signal score ($\ge 0.55$):

$$\text{occlusion\_score} = 0.30 \cdot S_{\text{overlap}} + 0.30 \cdot S_{\text{depth}} + 0.15 \cdot S_{\text{boundary\_contact}} + 0.15 \cdot S_{\text{trust}} + 0.10 \cdot S_{\text{proximity}}$$

4. **Render Relevance Filtering**: Irrelevant edges (`LOW`, `IGNORE`) between background regions are filtered out to keep the graph sparse and actionable.

---

## 6. Diagnostic Outputs & Visual Debug Artifacts

The spatial engine exports diagnostic artifacts under `output/<short_hash>/`:
- `spatial_scene.json`: Raw vs rejected vs merged vs trusted entity counts, trust scores, semantic roles, and parts.
- `spatial_relationships.json`: Canonical sparse relationships with confidence and evidence breakdown.
- `consolidated_entities.png`: Color-coded entity bounding boxes and trust labels over the image.
- `depth_field.png`: Structured 2.5D rendering depth.
- `occlusion_map.png`: Boundary exposure risk map.
- `spatial_diagnostics.json`: Overall spatial confidence metrics.
