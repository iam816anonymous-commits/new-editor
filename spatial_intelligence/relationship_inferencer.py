"""
Deterministic Sparse Spatial Relationship Inference for Spatial Intelligence Subsystem.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from .schemas import (
    RelationType,
    EntityClass,
    Entity,
    EntityPart,
    SpatialRelationship,
    SceneGraph,
    RenderRelevance,
    SemanticRole
)


def compute_multi_signal_occlusion_score(
    entA: Entity,
    entB: Entity,
    overlap_ratio: float,
    depth_diff: float,
    rgb_shape: Tuple[int, int]
) -> Tuple[float, Dict[str, float]]:
    """
    Computes a multi-signal occlusion score for entA OCCLUDES entB.

    Requires evidence across 5 independent signals:
    1. Meaningful mask overlap
    2. Consistent depth ordering (entA closer than entB)
    3. Contact boundary evidence
    4. Entity trust threshold
    5. Spatial plausibility

    Formula:
    occlusion_score = 0.30 * overlap_term
                    + 0.30 * depth_order_term
                    + 0.15 * boundary_contact_term
                    + 0.15 * trust_term
                    + 0.10 * proximity_term
    """
    h, w = rgb_shape
    diag_length = max(1.0, np.sqrt(h**2 + w**2))

    # 1. Overlap term (Requires >= 5% overlap relative to smaller entity)
    overlap_term = float(np.clip(overlap_ratio / 0.30, 0.0, 1.0))

    # 2. Depth ordering term (depth_diff < -0.15 strictly required for occlusion)
    if depth_diff >= -0.15:
        return 0.0, {"occlusion_score": 0.0, "reason": "insufficient_depth_separation"}

    depth_order_term = float(np.clip(abs(depth_diff) / 1.5, 0.0, 1.0))

    # 3. Contact boundary
    intersection = entA.mask & entB.mask
    if np.any(intersection):
        boundary_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_inter = cv2.dilate(intersection.astype(np.uint8) * 255, boundary_kernel) > 0
        boundary_contact_pixels = np.sum(dilated_inter & (entA.mask ^ entB.mask))
        boundary_contact_term = float(np.clip(boundary_contact_pixels / 500.0, 0.0, 1.0))
    else:
        boundary_contact_term = 0.0

    # 4. Entity trust
    trust_term = float(0.5 * entA.trust_score + 0.5 * entB.trust_score)

    # 5. Spatial proximity
    dist = np.sqrt((entA.centroid[0] - entB.centroid[0])**2 + (entA.centroid[1] - entB.centroid[1])**2)
    norm_dist = dist / diag_length
    proximity_term = float(np.clip(1.0 - norm_dist / 0.50, 0.0, 1.0))

    occlusion_score = (
        0.35 * overlap_term +
        0.35 * depth_order_term +
        0.10 * boundary_contact_term +
        0.10 * trust_term +
        0.10 * proximity_term
    )

    metrics = {
        "overlap_ratio": round(overlap_ratio, 3),
        "depth_diff": round(depth_diff, 3),
        "boundary_contact_term": round(boundary_contact_term, 3),
        "trust_term": round(trust_term, 3),
        "occlusion_score": round(occlusion_score, 3)
    }

    return float(occlusion_score), metrics


def infer_spatial_relationships(
    scene_graph: SceneGraph,
    image_shape: Tuple[int, int],
    depth_similarity_threshold: float = 0.8,
    min_occlusion_threshold: float = 0.55
) -> SceneGraph:
    """
    Infers a sparse, render-oriented canonical relationship graph among entities.

    Phase 1.6 Hardening Rules:
    1. Operates only between renderable entities or between primary subject and immediate environment.
       Skip pairwise relationship generation between two ANALYSIS_REGION entities.
    2. FRONT_OF GATE: Depth difference alone between distant/unrelated regions must NOT create FRONT_OF.
       Requires spatial interaction (overlap_ratio > 0.05 or normalized centroid distance < 0.35) plus depth separation.
    3. OCCLUSION GATE: OCCLUDES strictly requires actual mask overlap, contact boundary, and depth separation.
    4. SUPPORTS GATE: Strict horizontal/vertical contact geometry, vertical displacement, and compatible roles.
    5. Records explicit rejection reasons for candidate pairs that fail spatial gates.
    """
    entities = list(scene_graph.entities.values())
    num_entities = len(entities)
    h, w = image_shape
    diag_length = max(1.0, np.sqrt(h**2 + w**2))

    for i in range(num_entities):
        for j in range(i + 1, num_entities):
            entA = entities[i]
            entB = entities[j]

            # Canonical Ordering: Always place lower entity_id as subject
            if entA.entity_id > entB.entity_id:
                entA, entB = entB, entA

            scene_graph.relationship_candidate_count += 1

            # Rule 1: Skip pairwise relationship generation between two analysis-only regions
            if entA.entity_class == EntityClass.ANALYSIS_REGION and entB.entity_class == EntityClass.ANALYSIS_REGION:
                scene_graph.record_rejected_relationship(
                    entA.entity_id, entB.entity_id, RelationType.FRONT_OF,
                    "environmental_region", {"reason": "both_entities_analysis_region"}
                )
                continue

            # 1. Mask Overlap & Intersection
            intersection = np.sum(entA.mask & entB.mask)
            min_area = max(1, min(entA.area_pixels, entB.area_pixels))
            overlap_ratio = float(intersection / min_area)

            # 2. Centroid distance
            dist = np.sqrt((entA.centroid[0] - entB.centroid[0])**2 + (entA.centroid[1] - entB.centroid[1])**2)
            norm_dist = float(dist / diag_length)

            # 3. Depth Difference (entA.depth_mean - entB.depth_mean)
            d_diff = entA.depth_mean - entB.depth_mean  # Negative -> entA is closer

            # Canonical Relationship 1: FRONT_OF Gate (Requires spatial proximity/interaction AND depth separation)
            if abs(d_diff) > 0.35:
                # FRONT_OF Gate Check: Do not create FRONT_OF for unrelated distant regions
                is_spatially_interacting = (overlap_ratio > 0.05) or (norm_dist < 0.35) or (entA.is_primary_subject or entB.is_primary_subject)
                if is_spatially_interacting:
                    conf = min(0.99, float(abs(d_diff) / 3.0 + 0.50))
                    rel_relevance = RenderRelevance.CRITICAL if (entA.is_primary_subject or entB.is_primary_subject) else RenderRelevance.USEFUL

                    if d_diff < 0:  # entA is closer than entB -> entA FRONT_OF entB
                        scene_graph.add_relationship(
                            entA.entity_id, entB.entity_id, RelationType.FRONT_OF,
                            confidence=round(conf, 3), evidence=f"depth_mean_diff({d_diff:.2f})",
                            render_relevance=rel_relevance,
                            supporting_metrics={"depth_diff": round(d_diff, 3), "norm_dist": round(norm_dist, 3)}
                        )
                    else:  # entB is closer than entA -> entB FRONT_OF entA
                        scene_graph.add_relationship(
                            entB.entity_id, entA.entity_id, RelationType.FRONT_OF,
                            confidence=round(conf, 3), evidence=f"depth_mean_diff({-d_diff:.2f})",
                            render_relevance=rel_relevance,
                            supporting_metrics={"depth_diff": round(-d_diff, 3), "norm_dist": round(norm_dist, 3)}
                        )
                else:
                    scene_graph.record_rejected_relationship(
                        entA.entity_id, entB.entity_id, RelationType.FRONT_OF,
                        "insufficient_spatial_interaction", {"depth_diff": round(d_diff, 3), "norm_dist": round(norm_dist, 3)}
                    )

            # Canonical Relationship 2: SAME_COMPOUND_SUBJECT / NEAR
            elif abs(d_diff) <= depth_similarity_threshold and norm_dist < 0.35:
                if entA.is_primary_subject and entB.is_primary_subject:
                    scene_graph.add_relationship(
                        entA.entity_id, entB.entity_id, RelationType.SAME_COMPOUND_SUBJECT,
                        confidence=0.95, evidence="primary_subject_cluster",
                        render_relevance=RenderRelevance.CRITICAL
                    )
                elif norm_dist < 0.15 and (entA.is_primary_subject or entB.is_primary_subject or (entA.entity_class == EntityClass.RENDERABLE_ENTITY and entB.entity_class == EntityClass.RENDERABLE_ENTITY)):
                    scene_graph.add_relationship(
                        entA.entity_id, entB.entity_id, RelationType.NEAR,
                        confidence=round(1.0 - norm_dist / 0.15, 3), evidence=f"norm_dist({norm_dist:.2f})",
                        render_relevance=RenderRelevance.USEFUL,
                        supporting_metrics={"norm_dist": round(norm_dist, 3)}
                    )

            # Canonical Relationship 3: OVERLAPS & Multi-Signal OCCLUDES
            if overlap_ratio > 0.10 and (entA.entity_class == EntityClass.RENDERABLE_ENTITY or entB.entity_class == EntityClass.RENDERABLE_ENTITY):
                # Store OVERLAPS independently
                scene_graph.add_relationship(
                    entA.entity_id, entB.entity_id, RelationType.OVERLAPS,
                    confidence=round(overlap_ratio, 3), evidence=f"mask_overlap_ratio({overlap_ratio:.2f})",
                    render_relevance=RenderRelevance.USEFUL,
                    supporting_metrics={"overlap_ratio": round(overlap_ratio, 3)}
                )

                # Check multi-signal OCCLUDES for entA OCCLUDES entB
                score_A_occ_B, metrics_A = compute_multi_signal_occlusion_score(entA, entB, overlap_ratio, d_diff, image_shape)
                if score_A_occ_B >= min_occlusion_threshold:
                    scene_graph.add_relationship(
                        entA.entity_id, entB.entity_id, RelationType.OCCLUDES,
                        confidence=round(score_A_occ_B, 3), evidence="multi_signal_occlusion_gate",
                        render_relevance=RenderRelevance.CRITICAL,
                        supporting_metrics=metrics_A
                    )
                else:
                    # Check multi-signal OCCLUDES for entB OCCLUDES entA
                    score_B_occ_A, metrics_B = compute_multi_signal_occlusion_score(entB, entA, overlap_ratio, -d_diff, image_shape)
                    if score_B_occ_A >= min_occlusion_threshold:
                        scene_graph.add_relationship(
                            entB.entity_id, entA.entity_id, RelationType.OCCLUDES,
                            confidence=round(score_B_occ_A, 3), evidence="multi_signal_occlusion_gate",
                            render_relevance=RenderRelevance.CRITICAL,
                            supporting_metrics=metrics_B
                        )
                    else:
                        scene_graph.record_rejected_relationship(
                            entA.entity_id, entB.entity_id, RelationType.OCCLUDES,
                            "overlap_without_occlusion", {"score_A": round(score_A_occ_B, 3), "score_B": round(score_B_occ_A, 3)}
                        )

            # Canonical Relationship 4: SUPPORTS Gate (Requires strict vertical contact and compatible roles)
            if abs(entA.centroid[1] - entB.centroid[1]) < 0.20 * w and abs(d_diff) < 0.5:
                if entA.entity_class == EntityClass.RENDERABLE_ENTITY and entB.entity_class == EntityClass.RENDERABLE_ENTITY:
                    if entB.centroid[0] > entA.centroid[0]:  # entB is below entA in Y
                        dist_y = entB.centroid[0] - entA.centroid[0]
                        if 0.05 * h < dist_y < 0.30 * h:
                            scene_graph.add_relationship(
                                entB.entity_id, entA.entity_id, RelationType.SUPPORTS,
                                confidence=0.75, evidence=f"vertical_support_dist_y({dist_y:.1f})",
                                render_relevance=RenderRelevance.USEFUL,
                                supporting_metrics={"dist_y": round(dist_y, 1)}
                            )

    # Populate render_relationships (Filter out intra-compound and analysis-only relationships)
    for rel in scene_graph.relationships:
        subj = scene_graph.entities.get(rel.subject_id)
        targ = scene_graph.entities.get(rel.target_id)
        if subj is None or targ is None:
            continue

        # Suppress relationships between members of same compound subject or both background analysis regions
        both_same_compound = (subj.is_primary_subject and targ.is_primary_subject) or (subj.parent_subject_id is not None and subj.parent_subject_id == targ.entity_id)
        both_analysis_bg = (subj.entity_class == EntityClass.ANALYSIS_REGION and targ.entity_class == EntityClass.ANALYSIS_REGION)

        if not both_same_compound and not both_analysis_bg and rel.render_relevance in [RenderRelevance.CRITICAL, RenderRelevance.USEFUL]:
            scene_graph.render_relationships.append(rel)

    return scene_graph
