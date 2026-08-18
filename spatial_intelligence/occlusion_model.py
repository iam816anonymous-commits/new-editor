"""
Explicit Occlusion Model and Disocclusion Exposure Reasoning.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from .schemas import (
    OcclusionRelationship,
    SceneGraph,
    RelationType,
    Entity
)


def build_occlusion_model(
    scene_graph: SceneGraph,
    image_shape: Tuple[int, int]
) -> List[OcclusionRelationship]:
    """
    Constructs explicit occlusion relationships between pairs of entities that overlap
    and exhibit depth separation.
    """
    occlusion_relationships: List[OcclusionRelationship] = []
    h, w = image_shape

    for rel in scene_graph.relationships:
        if rel.relation_type == RelationType.OCCLUDES:
            occluder = scene_graph.entities.get(rel.subject_id)
            occluded = scene_graph.entities.get(rel.target_id)

            if occluder is not None and occluded is not None:
                # Boundary mask between occluder and occluded
                overlap_mask = occluder.mask & occluded.mask

                # Dilate overlap boundary to capture potential disocclusion exposure zone
                boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                dilated_overlap = cv2.dilate((overlap_mask.astype(np.uint8)) * 255, boundary_kernel) > 0
                disocclusion_risk_area = int(np.sum(dilated_overlap))

                occlusion_relationships.append(OcclusionRelationship(
                    occluder_id=occluder.entity_id,
                    occluded_id=occluded.entity_id,
                    boundary_mask=dilated_overlap,
                    disocclusion_risk_area=disocclusion_risk_area,
                    confidence=rel.confidence
                ))

    return occlusion_relationships


def generate_occlusion_map(
    occlusion_relationships: List[OcclusionRelationship],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """Generates an occlusion risk map visualization in range [0..255]."""
    h, w = image_shape
    occlusion_vis = np.zeros((h, w), dtype=np.uint8)

    for occ in occlusion_relationships:
        conf_byte = int(occ.confidence * 255.0)
        occlusion_vis[occ.boundary_mask] = np.maximum(occlusion_vis[occ.boundary_mask], conf_byte)

    return occlusion_vis
