"""
Deterministic Entity Trust Scoring and Semantic Role Classification.
Calculates entity_trust_score separate from raw SAM segmentation confidence.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from .schemas import (
    Entity,
    EntityClass,
    EntityTrustScore,
    SemanticRole,
    RenderRelevance
)


def compute_entity_trust_score(
    entity: Entity,
    rgb_array: np.ndarray,
    refined_depth: np.ndarray,
    confidence_map: np.ndarray
) -> EntityTrustScore:
    """
    Computes a deterministic, explainable entity_trust_score in range [0.0, 1.0].

    Formula:
    entity_trust_score = 0.25 * mask_quality
                       + 0.25 * depth_coherence
                       + 0.20 * boundary_coherence
                       + 0.15 * candidate_uniqueness
                       + 0.15 * render_relevance

    Note: An entity can have a high SAM segmentation score while having low trust if it is
    a duplicate, an environmental blob, or geometrically incoherent.
    """
    h, w, _ = rgb_array.shape
    total_pixels = h * w
    mask = entity.mask

    if not np.any(mask):
        return EntityTrustScore(
            mask_quality=0.0,
            depth_coherence=0.0,
            boundary_coherence=0.0,
            candidate_uniqueness=0.0,
            render_relevance=0.0,
            overall_trust_score=0.0,
            evidence_breakdown={}
        )

    # 1. Mask Quality (Size ratio, compactness, non-fragmentation)
    area_ratio = entity.area_ratio
    bbox = entity.bbox
    bbox_area = max(1, (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1))
    compactness = float(np.clip(entity.area_pixels / bbox_area, 0.0, 1.0))

    # Scale score: preferred area coverage in [0.01, 0.60]
    if 0.01 <= area_ratio <= 0.60:
        scale_score = 1.0
    elif area_ratio < 0.01:
        scale_score = area_ratio / 0.01
    else:
        scale_score = max(0.0, (0.85 - area_ratio) / 0.25)

    mask_quality = float(0.5 * compactness + 0.5 * scale_score)

    # 2. Depth Coherence (Low std relative to span, distinct from surroundings)
    d_min, d_max = float(refined_depth.min()), float(refined_depth.max())
    d_span = max(1e-5, d_max - d_min)

    # Std term: smaller depth std inside mask -> higher coherence
    norm_std = entity.depth_std / d_span
    depth_std_score = float(np.clip(1.0 - norm_std / 0.3, 0.0, 1.0))

    # Surrounding ring depth separation
    dilated = cv2.dilate((mask.astype(np.uint8)) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
    surrounding_ring = dilated & (~mask)
    if np.any(surrounding_ring):
        surr_depth_mean = float(np.mean(refined_depth[surrounding_ring]))
        depth_sep = abs(entity.depth_mean - surr_depth_mean) / d_span
        depth_sep_score = float(np.clip(depth_sep / 0.15, 0.0, 1.0))
    else:
        depth_sep_score = 0.5

    depth_coherence = float(0.5 * depth_std_score + 0.5 * depth_sep_score)

    # 3. Boundary Coherence (Gradient alignment with RGB color boundaries)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    boundary = cv2.Canny((mask.astype(np.uint8)) * 255, 100, 200) > 0

    if np.any(boundary):
        dilated_edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0
        boundary_alignment = float(np.mean(dilated_edges[boundary]))
        # Mean depth confidence map value along boundary
        conf_at_boundary = float(np.mean(confidence_map[boundary]))
        boundary_coherence = float(0.5 * boundary_alignment + 0.5 * conf_at_boundary)
    else:
        boundary_coherence = 0.5

    # 4. Candidate Uniqueness
    num_sources = len(entity.source_candidate_ids)
    candidate_uniqueness = 1.0 if num_sources == 1 else float(np.clip(1.0 - (num_sources - 1) * 0.1, 0.5, 1.0))

    # 5. Render Relevance
    if entity.is_primary_subject:
        render_relevance = 1.0
    else:
        # Distance to center
        cy, cx = entity.norm_centroid
        dist_to_center = np.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
        centrality = float(np.clip(1.0 - dist_to_center / 0.5, 0.0, 1.0))

        # Depth saliency: closer depth Z -> higher render relevance
        saliency = float(1.0 - (entity.depth_mean - d_min) / d_span)
        render_relevance = float(0.5 * centrality + 0.5 * saliency)

    # Weighted Overall Score
    overall_trust = (
        0.25 * mask_quality +
        0.25 * depth_coherence +
        0.20 * boundary_coherence +
        0.15 * candidate_uniqueness +
        0.15 * render_relevance
    )

    evidence = {
        "mask_quality": round(mask_quality, 4),
        "depth_coherence": round(depth_coherence, 4),
        "boundary_coherence": round(boundary_coherence, 4),
        "candidate_uniqueness": round(candidate_uniqueness, 4),
        "render_relevance": round(render_relevance, 4)
    }

    return EntityTrustScore(
        mask_quality=round(mask_quality, 4),
        depth_coherence=round(depth_coherence, 4),
        boundary_coherence=round(boundary_coherence, 4),
        candidate_uniqueness=round(candidate_uniqueness, 4),
        render_relevance=round(render_relevance, 4),
        overall_trust_score=round(overall_trust, 4),
        evidence_breakdown=evidence
    )


def classify_entity_class_role_and_relevance(
    entity: Entity,
    depth_span_min: float,
    depth_span_max: float
) -> Tuple[EntityClass, SemanticRole, RenderRelevance]:
    """
    Classifies entity into EntityClass (RENDERABLE_ENTITY vs ANALYSIS_REGION), coarse SemanticRole, and RenderRelevance.

    Environmental Suppression Rules:
    - PRIMARY_SUBJECT is ALWAYS RENDERABLE_ENTITY and CRITICAL.
    - Large background regions (>25% area) with low boundary alignment or high depth variance are classified as
      ANALYSIS_REGION and IGNORE/LOW render relevance so they do NOT create competing renderable objects or pairwise graph clutter.
    - Compact foreground/midground objects with strong trust scores remain RENDERABLE_ENTITY.
    """
    if entity.is_primary_subject:
        return EntityClass.RENDERABLE_ENTITY, SemanticRole.PRIMARY_SUBJECT, RenderRelevance.CRITICAL

    d_span = max(1e-5, depth_span_max - depth_span_min)
    norm_depth = (entity.depth_mean - depth_span_min) / d_span

    # Multi-signal check for analysis-only background regions
    is_large_background = entity.area_ratio > 0.25
    is_deep_background = norm_depth > 0.60
    has_low_trust = entity.trust_score < 0.65

    if is_large_background or (is_deep_background and has_low_trust):
        ent_class = EntityClass.ANALYSIS_REGION
        role = SemanticRole.BACKGROUND_REGION if entity.area_ratio > 0.15 else SemanticRole.EMPTY_BACKGROUND
        relevance = RenderRelevance.IGNORE
    elif norm_depth < 0.25 and entity.trust_score > 0.50:
        ent_class = EntityClass.RENDERABLE_ENTITY
        role = SemanticRole.FOREGROUND_OBJECT
        relevance = RenderRelevance.CRITICAL if entity.trust_score > 0.70 else RenderRelevance.USEFUL
    elif norm_depth < 0.60 and entity.trust_score > 0.55 and entity.area_ratio < 0.25:
        ent_class = EntityClass.RENDERABLE_ENTITY
        role = SemanticRole.MIDGROUND_OBJECT
        relevance = RenderRelevance.USEFUL
    else:
        # Default fallback for uninformative/weak environmental blobs
        ent_class = EntityClass.ANALYSIS_REGION
        role = SemanticRole.SECONDARY_OBJECT if norm_depth < 0.50 else SemanticRole.BACKGROUND_REGION
        relevance = RenderRelevance.LOW

    return ent_class, role, relevance


def process_entity_trust_and_roles(
    entities: List[Entity],
    rgb_array: np.ndarray,
    refined_depth: np.ndarray,
    confidence_map: np.ndarray
) -> List[Entity]:
    """Computes trust score, entity class, and semantic role for all entities in list."""
    d_min, d_max = float(refined_depth.min()), float(refined_depth.max())

    for ent in entities:
        trust_details = compute_entity_trust_score(ent, rgb_array, refined_depth, confidence_map)
        ent.trust_details = trust_details
        ent.trust_score = trust_details.overall_trust_score

        ent_class, role, relevance = classify_entity_class_role_and_relevance(ent, d_min, d_max)
        ent.entity_class = ent_class
        ent.semantic_role = role
        ent.render_relevance = relevance

    return entities
