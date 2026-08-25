# Phase 3.4 — Subject Anchor Representation & Data Model

## Executive Summary
Phase 3.4 introduces the `SubjectAnchor` model to represent the primary subject as a coherent 3D visual object during 2.5D view synthesis, preventing monocular depth noise from inducing independent per-pixel drift or texture swimming.

## Data Model Definition
Defined in `modes/mode_2_5d/scene.py`:

```python
@dataclass
class SubjectAnchor:
    """Anchor representation enforcing rigid temporal stability for primary subject."""
    mask: np.ndarray
    soft_mask: np.ndarray
    bounding_box: Tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax)
    centroid: Tuple[float, float]
    reference_depth: float
    scale: float
    orientation: float
    confidence: float
```

## Extraction & Binding
`extract_subject_anchor` derives the subject anchor directly from SAM 2 segmentation masks and refined Depth Anything V2 depth fields:
- **Reference Depth**: Extracted median depth inside subject mask $Z_{\text{ref}} = \text{median}(Z_{\text{subject}})$.
- **Centroid & Scale**: Geometric center $(c_y, c_x)$ and scale parameter $\sqrt{\text{Area}}$.
- **Soft Mask**: Distance transform falloff $\text{dist}(u,v) / \max(\text{dist})$ providing soft-boundary blending.

The `SubjectAnchor` is attached directly to the `SceneLayer` for `PRIMARY_SUBJECT` in `LayeredScene`.
