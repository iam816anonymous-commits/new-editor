"""
Deterministic Candidate Consolidation Stage for Spatial Intelligence Subsystem.
Transforms raw segmentation candidates into consolidated trusted scene entities.
"""

from typing import List, Dict, Tuple, Set, Optional, Any
import numpy as np
from .schemas import (
    SegmentationCandidate,
    Entity,
    EntityPart,
    SemanticRole,
    RenderRelevance,
    EntityTrustScore
)


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two 2D boolean masks."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return float(intersection / union) if union > 0 else 0.0


def compute_mask_containment(child_mask: np.ndarray, parent_mask: np.ndarray) -> float:
    """Computes fraction of child_mask contained inside parent_mask."""
    child_area = np.sum(child_mask)
    if child_area == 0:
        return 0.0
    intersection = np.sum(child_mask & parent_mask)
    return float(intersection / child_area)


def consolidate_candidates(
    raw_candidates: List[SegmentationCandidate],
    primary_subject_mask: Optional[np.ndarray],
    primary_subject_group_ids: List[int],
    refined_depth: np.ndarray,
    rgb_shape: Tuple[int, int],
    iou_merge_threshold: float = 0.70,
    containment_merge_threshold: float = 0.85,
    max_trusted_environmental_entities: int = 12
) -> Tuple[List[Entity], int, int]:
    """
    Consolidates raw SAM 2/grid segmentation candidates into a small set of trusted SpatialEntities.

    Rules:
    1. Primary Subject Protection: Primary subject is preserved as Entity 1 without fragmenting.
       Candidates that overlap >60% with primary subject are consolidated into the primary subject's source candidate list.
    2. Containment & Duplicate Merging: Non-primary candidates with IoU > iou_merge_threshold or
       containment > containment_merge_threshold and compatible depth are merged.
    3. Noise & Enormous Mask Rejection: Tiny candidates (<0.5% area) or huge environmental masks (>85% area)
       that lack distinctive depth boundaries are rejected.

    Returns:
        (consolidated_entities, rejected_count, merged_count)
    """
    h, w = rgb_shape
    total_pixels = max(1, h * w)

    rejected_count = 0
    merged_count = 0

    entities: List[Entity] = []

    # 1. Primary Subject Entity Creation
    if primary_subject_mask is not None and np.any(primary_subject_mask):
        sub_y, sub_x = np.where(primary_subject_mask)
        ymin, ymax = int(np.min(sub_y)), int(np.max(sub_y))
        xmin, xmax = int(np.min(sub_x)), int(np.max(sub_x))
        sub_bbox = (ymin, xmin, ymax, xmax)
        sub_cy, sub_cx = float(np.mean(sub_y)), float(np.mean(sub_x))

        sub_depths = refined_depth[primary_subject_mask]
        sub_area = int(np.sum(primary_subject_mask))

        primary_entity = Entity(
            entity_id=1,
            name="primary_compound_subject",
            mask=primary_subject_mask,
            bbox=sub_bbox,
            centroid=(sub_cy, sub_cx),
            norm_centroid=(sub_cy / max(1, h), sub_cx / max(1, w)),
            area_pixels=sub_area,
            area_ratio=float(sub_area / total_pixels),
            depth_mean=float(np.mean(sub_depths)),
            depth_median=float(np.median(sub_depths)),
            depth_std=float(np.std(sub_depths)),
            semantic_role=SemanticRole.PRIMARY_SUBJECT,
            trust_score=1.0,
            is_primary_subject=True,
            source_candidate_ids=list(primary_subject_group_ids),
            render_relevance=RenderRelevance.CRITICAL,
            parts=[]
        )
        entities.append(primary_entity)

    # Filter out candidates already absorbed into primary subject or invalid
    environmental_candidates: List[SegmentationCandidate] = []
    for cand in raw_candidates:
        if cand.candidate_id in primary_subject_group_ids:
            continue

        # Check coverage
        cov = cand.area_ratio

        # Reject tiny fragments (< 0.5% image area)
        if cov < 0.005:
            rejected_count += 1
            continue

        # Reject huge uninformative masks (>98% total image area)
        if cov > 0.98:
            rejected_count += 1
            continue

        # Check overlap with primary subject
        if primary_subject_mask is not None and np.any(primary_subject_mask):
            sub_containment = compute_mask_containment(cand.mask, primary_subject_mask)
            if sub_containment > 0.60:
                # Absorb candidate into primary subject candidate provenance
                if entities and entities[0].is_primary_subject:
                    entities[0].source_candidate_ids.append(cand.candidate_id)
                merged_count += 1
                continue

        environmental_candidates.append(cand)

    # Sort environmental candidates by SAM confidence * depth saliency descending
    environmental_candidates.sort(key=lambda c: c.sam_confidence, reverse=True)

    # Conservative Compound-Subject Collapsing Pass: Group proposals with high mask overlap (>50%), similar depth (|delta Z| < 1.0)
    clusters: List[List[SegmentationCandidate]] = []

    for cand in environmental_candidates:
        assigned_cluster = False
        for cluster in clusters:
            rep = cluster[0]  # Cluster representative

            iou = compute_mask_iou(cand.mask, rep.mask)
            containment_A_in_B = compute_mask_containment(cand.mask, rep.mask)
            containment_B_in_A = compute_mask_containment(rep.mask, cand.mask)
            depth_diff = abs(cand.depth_mean - rep.depth_mean)

            # Check if candidates represent substantially the same spatial region or compound part
            if (iou >= 0.50 or containment_A_in_B >= 0.70 or containment_B_in_A >= 0.70) and depth_diff < 1.2:
                cluster.append(cand)
                assigned_cluster = True
                merged_count += 1
                break

        if not assigned_cluster:
            clusters.append([cand])

    # Convert top clusters into SpatialEntities
    ent_id_counter = 2
    for cluster in clusters:
        if len(entities) - 1 >= max_trusted_environmental_entities:
            rejected_count += len(cluster)
            continue

        # Primary representative is the one with highest SAM confidence
        rep = max(cluster, key=lambda c: c.sam_confidence)

        # Merge masks of cluster members
        merged_mask = np.zeros((h, w), dtype=bool)
        source_ids = []
        for member in cluster:
            merged_mask |= member.mask
            source_ids.append(member.candidate_id)

        # Ensure no overlap with primary subject mask
        if primary_subject_mask is not None:
            merged_mask &= (~primary_subject_mask)

        merged_area = int(np.sum(merged_mask))
        if merged_area < 100:  # Skip if residual mask is trivial
            rejected_count += len(cluster)
            continue

        y_indices, x_indices = np.where(merged_mask)
        bbox = (int(np.min(y_indices)), int(np.min(x_indices)), int(np.max(y_indices)), int(np.max(x_indices)))
        cy, cx = float(np.mean(y_indices)), float(np.mean(x_indices))

        sec_depths = refined_depth[merged_mask]

        entities.append(Entity(
            entity_id=ent_id_counter,
            name=f"spatial_entity_{ent_id_counter}",
            mask=merged_mask,
            bbox=bbox,
            centroid=(cy, cx),
            norm_centroid=(cy / max(1, h), cx / max(1, w)),
            area_pixels=merged_area,
            area_ratio=float(merged_area / total_pixels),
            depth_mean=float(np.mean(sec_depths)),
            depth_median=float(np.median(sec_depths)),
            depth_std=float(np.std(sec_depths)),
            semantic_role=SemanticRole.SECONDARY_OBJECT,  # Will be classified in entity_trust module
            trust_score=rep.sam_confidence,
            is_primary_subject=False,
            source_candidate_ids=source_ids,
            render_relevance=RenderRelevance.USEFUL,
            parts=[]
        ))
        ent_id_counter += 1

    return entities, rejected_count, merged_count
