# Phase 3.3 — 2.5D Layered Scene Representation & Data Model

## Executive Summary
Phase 3.3 introduces the formal `LayeredScene` and `SceneLayer` data architecture to support multi-layer depth-decomposed rendering for 2.5D view synthesis. This replaces simple single-plate background assumptions with structured semantic layers (`BACKGROUND`, `MIDGROUND`, `PRIMARY_SUBJECT`, `FOREGROUND`), each tracking independent texture, depth, valid masks, and inpaint masks.

## Data Structures
Located in `modes/mode_2_5d/scene.py`:

```python
@dataclass
class SceneLayer:
    layer_id: str
    semantic_role: str  # BACKGROUND, MIDGROUND, PRIMARY_SUBJECT, FOREGROUND
    rgb: np.ndarray
    alpha: np.ndarray
    depth: np.ndarray
    valid_mask: np.ndarray
    inpaint_mask: np.ndarray
    z_min: float
    z_max: float
    priority: int

@dataclass
class LayeredScene:
    layers: List[SceneLayer]
    source_image: np.ndarray
    refined_depth: np.ndarray
    subject_mask: np.ndarray
    occlusion_edge_map: np.ndarray
    convergence_depth: float
    metadata: Dict[str, Any]
```

## Layer Construction & Priority
`construct_layered_scene` automatically segments an input image and refined depth map into ordered layers:
1. `BACKGROUND` (Priority 0): Includes background plate, extrapolated depth, and occlusion boundary inpaint flags.
2. `PRIMARY_SUBJECT` (Priority 1): Rigid core subject isolated via binary alpha mask with preserved local geometry.

## Integration
The layered scene representation feeds directly into `Renderer25D.render_layered_scene` to perform multi-pass Z-buffered forward splatting with correct depth composition order.
