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

    return {
        "entities_count": len(scene_graph.entities),
        "relationships_count": len(scene_graph.relationships),
        "entities": entities_export,
        "relationships": relationships_export
    }
