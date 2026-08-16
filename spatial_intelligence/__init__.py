"""
Spatial Intelligence Subsystem for First-Principles Cinematic 2.5D Parallax Renderer (V0).
"""

from .schemas import (
    RelationType,
    SemanticRole,
    RenderRelevance,
    SegmentationCandidate,
    EntityTrustScore,
    EntityPart,
    Entity,
    SpatialRelationship,
    SceneGraph,
    DepthField,
    OcclusionRelationship,
    CameraModel,
    SpatialConfidence,
    SpatialDiagnostics
)
from .scene_graph import create_scene_graph, export_scene_graph_dict
from .entity_consolidator import consolidate_candidates
from .entity_trust import compute_entity_trust_score, classify_semantic_role_and_relevance, process_entity_trust_and_roles
from .relationship_inferencer import infer_spatial_relationships

__all__ = [
    "RelationType",
    "SemanticRole",
    "RenderRelevance",
    "SegmentationCandidate",
    "EntityTrustScore",
    "EntityPart",
    "Entity",
    "SpatialRelationship",
    "SceneGraph",
    "DepthField",
    "OcclusionRelationship",
    "CameraModel",
    "SpatialConfidence",
    "SpatialDiagnostics",
    "create_scene_graph",
    "export_scene_graph_dict",
    "consolidate_candidates",
    "compute_entity_trust_score",
    "classify_semantic_role_and_relevance",
    "process_entity_trust_and_roles",
    "infer_spatial_relationships"
]
