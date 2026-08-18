"""
Scene Graph Construction and Management for Spatial Intelligence Subsystem.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from .schemas import (
    RelationType,
    EntityPart,
    Entity,
    SpatialRelationship,
    SceneGraph
)


def create_scene_graph(entities: List[Entity]) -> SceneGraph:
    """Creates a new SceneGraph initialized with the given list of entities."""
    sg = SceneGraph()
    for ent in entities:
        sg.add_entity(ent)
    return sg


def export_scene_graph_dict(scene_graph: SceneGraph) -> Dict[str, Any]:
    """Exports scene graph structure into JSON-serializable dictionary."""
    entities_export = []
    for ent in scene_graph.entities.values():
        parts_export = [
            {
                "part_id": p.part_id,
                "entity_id": p.entity_id,
                "name": p.name,
                "bbox": p.bbox,
                "depth_mean": round(p.depth_mean, 3),
                "depth_std": round(p.depth_std, 3),
                "confidence": round(p.confidence, 3)
            }
            for p in ent.parts
        ]
        entities_export.append({
            "entity_id": ent.entity_id,
            "name": ent.name,
            "bbox": ent.bbox,
            "centroid": [round(c, 2) for c in ent.centroid],
            "norm_centroid": [round(c, 3) for c in ent.norm_centroid],
            "area_pixels": ent.area_pixels,
            "area_ratio": round(ent.area_ratio, 4),
            "depth_mean": round(ent.depth_mean, 3),
            "depth_median": round(ent.depth_median, 3),
            "depth_std": round(ent.depth_std, 3),
            "is_primary_subject": ent.is_primary_subject,
            "parts": parts_export
        })

    relationships_export = [
        {
            "subject_id": r.subject_id,
            "target_id": r.target_id,
            "relation_type": r.relation_type.value,
            "confidence": round(r.confidence, 3),
            "evidence": r.evidence
        }
        for r in scene_graph.relationships
    ]

    rejected_export = [
        {
            "subject_id": rr.subject_id,
            "target_id": rr.target_id,
            "relation_type": rr.relation_type.value,
            "rejection_reason": rr.rejection_reason,
            "supporting_metrics": rr.supporting_metrics
        }
        for rr in scene_graph.rejected_relationships
    ]

    front_of_cnt = len([r for r in scene_graph.relationships if r.relation_type == RelationType.FRONT_OF])
    overlap_cnt = len([r for r in scene_graph.relationships if r.relation_type == RelationType.OVERLAPS])
    occlusion_cnt = len([r for r in scene_graph.relationships if r.relation_type == RelationType.OCCLUDES])
    support_cnt = len([r for r in scene_graph.relationships if r.relation_type == RelationType.SUPPORTS])

    return {
        "raw_candidate_count": scene_graph.raw_candidate_count,
        "rejected_candidate_count": scene_graph.rejected_candidate_count,
        "merged_candidate_count": scene_graph.merged_candidate_count,
        "trusted_entities_count": len(scene_graph.entities),
        "analysis_only_entity_count": scene_graph.analysis_only_entity_count,
        "renderable_entity_count": scene_graph.renderable_entity_count,
        "relationship_candidate_count": scene_graph.relationship_candidate_count,
        "relationship_rejected_count": scene_graph.relationship_rejected_count,
        "final_relationship_count": len(scene_graph.relationships),
        "front_of_count": front_of_cnt,
        "overlap_count": overlap_cnt,
        "occlusion_count": occlusion_cnt,
        "support_count": support_cnt,
        "entities": entities_export,
        "relationships": relationships_export,
        "rejected_relationships": rejected_export
    }
