"""
Typed Dataclasses and Contracts for Spatial Intelligence Subsystem.
First-Principles Cinematic 2.5D Parallax Renderer (V0)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class RelationType(str, Enum):
    """Enumeration of directed spatial relationships between entities/parts."""
    IN_FRONT_OF = "IN_FRONT_OF"
    BEHIND = "BEHIND"
    SUPPORTS = "SUPPORTS"
    ATTACHED_TO = "ATTACHED_TO"
    PART_OF = "PART_OF"
    OVERLAPS = "OVERLAPS"
    OCCLUDES = "OCCLUDES"
    OCCLUDED_BY = "OCCLUDED_BY"
    NEAR = "NEAR"
    FAR_FROM = "FAR_FROM"
    SAME_COMPOUND_SUBJECT = "SAME_COMPOUND_SUBJECT"


@dataclass
class EntityPart:
    """Sub-component part of a scene entity (e.g. face, arm, Shesha heads)."""
    part_id: int
    entity_id: int
    name: str
    mask: np.ndarray = field(repr=False)
    bbox: Tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax)
    depth_mean: float
    depth_std: float
    confidence: float


@dataclass
class Entity:
    """Semantically and spatially distinct scene entity."""
    entity_id: int
    name: str
    mask: np.ndarray = field(repr=False)
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    norm_centroid: Tuple[float, float]
    area_pixels: int
    area_ratio: float
    depth_mean: float
    depth_median: float
    depth_std: float
    is_primary_subject: bool = False
    parts: List[EntityPart] = field(default_factory=list)


@dataclass
class SpatialRelationship:
    """Directed spatial relationship edge between two entities or parts."""
    subject_id: int
    target_id: int
    relation_type: RelationType
    confidence: float
    evidence: str


@dataclass
class SceneGraph:
    """Graph representation holding scene entities, parts, and spatial relationship edges."""
    entities: Dict[int, Entity] = field(default_factory=dict)
    relationships: List[SpatialRelationship] = field(default_factory=list)

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.entity_id] = entity

    def add_relationship(self, subject_id: int, target_id: int, relation_type: RelationType, confidence: float, evidence: str) -> None:
        self.relationships.append(SpatialRelationship(subject_id, target_id, relation_type, confidence, evidence))

    def get_relationships_for_entity(self, entity_id: int) -> List[SpatialRelationship]:
        return [r for r in self.relationships if r.subject_id == entity_id or r.target_id == entity_id]


@dataclass
class DepthField:
    """Structured 2.5D spatial depth field representation."""
    rendering_depth: np.ndarray = field(repr=False)  # Z in [0.1, 10.0]
    background_depth: np.ndarray = field(repr=False)
    subject_depth: np.ndarray = field(repr=False)
    uncertainty_map: np.ndarray = field(repr=False)  # Uncertainty in [0.0, 1.0]
    provenance_map: np.ndarray = field(repr=False)   # 1.0 = OBSERVED, 0.0 = INFERRED/RECONSTRUCTED
    min_depth: float = 0.1
    max_depth: float = 10.0


@dataclass
class OcclusionRelationship:
    """Explicit occlusion relationship between occluder and occluded entity."""
    occluder_id: int
    occluded_id: int
    boundary_mask: np.ndarray = field(repr=False)
    disocclusion_risk_area: int
    confidence: float


@dataclass
class CameraModel:
    """Perspective pinhole camera model for 2.5D view synthesis."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))  # (tx, ty, tz)
    rotation_pitch_yaw_roll: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))  # (pitch, yaw, roll)

    def project_points(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Projects 3D points [X, Y, Z] to image coordinates (u', v') and depth Z'."""
        X = points_3d[:, 0]
        Y = points_3d[:, 1]
        Z = points_3d[:, 2]
        Z_safe = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
        u_proj = self.fx * (X / Z_safe) + self.cx
        v_proj = self.fy * (Y / Z_safe) + self.cy
        return u_proj, v_proj, Z


@dataclass
class SpatialConfidence:
    """Multi-layer spatial reasoning confidence scores."""
    relationship_confidence: float
    depth_field_confidence: float
    occlusion_confidence: float
    overall_spatial_confidence: float


@dataclass
class SpatialDiagnostics:
    """Complete exportable diagnostic payload for spatial intelligence subsystem."""
    scene_graph: SceneGraph
    depth_field: DepthField
    occlusion_relationships: List[OcclusionRelationship]
    camera_model: CameraModel
    spatial_confidence: SpatialConfidence
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
