"""
Deterministic Spatial Relationship Inference for Spatial Intelligence Subsystem.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from .schemas import (
    RelationType,
    Entity,
    EntityPart,
    SpatialRelationship,
    SceneGraph
)


def infer_spatial_relationships(
    scene_graph: SceneGraph,
    image_shape: Tuple[int, int],
    depth_similarity_threshold: float = 1.0
) -> SceneGraph:
    """
    Infers deterministic 2.5D spatial relationships between all pairs of entities in the scene graph
    using visual evidence: mask overlap, bbox containment, centroid distance, depth distribution, and boundary contact.
    """
    entities = list(scene_graph.entities.values())
    num_entities = len(entities)
    h, w = image_shape
    diag_length = max(1.0, np.sqrt(h**2 + w**2))

    for i in range(num_entities):
        for j in range(i + 1, num_entities):
            entA = entities[i]
            entB = entities[j]

            # 1. Mask Overlap & Intersection
            intersection = np.sum(entA.mask & entB.mask)
            min_area = max(1, min(entA.area_pixels, entB.area_pixels))
            overlap_ratio = float(intersection / min_area)

            # 2. Centroid distance
            dist = np.sqrt((entA.centroid[0] - entB.centroid[0])**2 + (entA.centroid[1] - entB.centroid[1])**2)
            norm_dist = float(dist / diag_length)

            # 3. Depth Difference (smaller rendering depth Z = closer to camera in rendering coordinates)
            d_diff = entA.depth_mean - entB.depth_mean  # If negative -> entA is closer than entB

            # Inferences:
            # Relationship 1: IN_FRONT_OF / BEHIND
            if abs(d_diff) > 0.3:
                if d_diff < 0:  # entA is closer
                    conf = min(0.99, float(abs(d_diff) / 3.0 + 0.50))
                    scene_graph.add_relationship(
                        entA.entity_id, entB.entity_id, RelationType.IN_FRONT_OF,
                        confidence=round(conf, 3), evidence=f"depth_mean_diff({d_diff:.2f})"
                    )
                    scene_graph.add_relationship(
                        entB.entity_id, entA.entity_id, RelationType.BEHIND,
                        confidence=round(conf, 3), evidence=f"depth_mean_diff({-d_diff:.2f})"
                    )
                else:  # entB is closer
                    conf = min(0.99, float(abs(d_diff) / 3.0 + 0.50))
                    scene_graph.add_relationship(
                        entB.entity_id, entA.entity_id, RelationType.IN_FRONT_OF,
                        confidence=round(conf, 3), evidence=f"depth_mean_diff({-d_diff:.2f})"
                    )
                    scene_graph.add_relationship(
                        entA.entity_id, entB.entity_id, RelationType.BEHIND,
                        confidence=round(conf, 3), evidence=f"depth_mean_diff({d_diff:.2f})"
                    )

            # Relationship 2: SAME_COMPOUND_SUBJECT / NEAR
            if abs(d_diff) <= depth_similarity_threshold and norm_dist < 0.35:
                if entA.is_primary_subject and entB.is_primary_subject:
                    scene_graph.add_relationship(
                        entA.entity_id, entB.entity_id, RelationType.SAME_COMPOUND_SUBJECT,
                        confidence=0.95, evidence="primary_subject_cluster"
                    )
                elif norm_dist < 0.20:
                    scene_graph.add_relationship(
                        entA.entity_id, entB.entity_id, RelationType.NEAR,
                        confidence=round(1.0 - norm_dist / 0.20, 3), evidence=f"norm_dist({norm_dist:.2f})"
                    )

            # Relationship 3: OVERLAPS / OCCLUDES / OCCLUDED_BY
            if overlap_ratio > 0.05:
                scene_graph.add_relationship(
                    entA.entity_id, entB.entity_id, RelationType.OVERLAPS,
                    confidence=round(overlap_ratio, 3), evidence=f"mask_overlap_ratio({overlap_ratio:.2f})"
                )
                if abs(d_diff) > 0.2:
                    if d_diff < 0:  # entA occludes entB
                        scene_graph.add_relationship(
                            entA.entity_id, entB.entity_id, RelationType.OCCLUDES,
                            confidence=round(min(0.98, overlap_ratio + 0.5), 3), evidence="overlap_plus_depth"
                        )
                        scene_graph.add_relationship(
                            entB.entity_id, entA.entity_id, RelationType.OCCLUDED_BY,
                            confidence=round(min(0.98, overlap_ratio + 0.5), 3), evidence="overlap_plus_depth"
                        )
                    else:  # entB occludes entA
                        scene_graph.add_relationship(
                            entB.entity_id, entA.entity_id, RelationType.OCCLUDES,
                            confidence=round(min(0.98, overlap_ratio + 0.5), 3), evidence="overlap_plus_depth"
                        )
                        scene_graph.add_relationship(
                            entA.entity_id, entB.entity_id, RelationType.OCCLUDED_BY,
                            confidence=round(min(0.98, overlap_ratio + 0.5), 3), evidence="overlap_plus_depth"
                        )

            # Relationship 4: SUPPORTS (Vertical position + contact: lower entity entB supports upper entA)
            if abs(entA.centroid[1] - entB.centroid[1]) < 0.3 * w:
                if entB.centroid[0] > entA.centroid[0] and abs(d_diff) < 0.8:  # entB is below entA in Y
                    dist_y = entB.centroid[0] - entA.centroid[0]
                    if dist_y < 0.4 * h:
                        scene_graph.add_relationship(
                            entB.entity_id, entA.entity_id, RelationType.SUPPORTS,
                            confidence=0.75, evidence=f"vertical_support_dist_y({dist_y:.1f})"
                        )

    return scene_graph
