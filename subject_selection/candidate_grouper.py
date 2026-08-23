"""
Candidate Grouping and Pairwise Compatibility Graph for Semantic Subject Selection.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from .schemas import CandidateFeatures, CandidateScore, CandidateGroup, SubjectSelectionConfig
from .candidate_features import extract_candidate_features
from .candidate_scorer import score_candidate_features
from .mask_merger import merge_candidate_masks, refine_subject_mask


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two 2D boolean masks."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return float(intersection / union) if union > 0 else 0.0


def compute_pairwise_compatibility(
    featA: CandidateFeatures,
    featB: CandidateFeatures,
    maskA: np.ndarray,
    maskB: np.ndarray,
    config: SubjectSelectionConfig = SubjectSelectionConfig()
) -> float:
    """
    Calculates pairwise compatibility score [0.0, 1.0] between two candidate masks.
    High compatibility indicates they likely belong to the same compound cinematic subject
    (e.g., Vishnu + Shesha), whereas low compatibility penalizes merging with unrelated background/planets.
    """
    # 1. Depth Similarity
    d_diff = abs(featA.foreground_depth_mean - featB.foreground_depth_mean)
    depth_compat = float(max(0.0, 1.0 - (d_diff / config.depth_similarity_threshold)))

    # 2. Spatial Proximity / Adjacency
    # Distance between centroids
    cent_dist = np.sqrt((featA.centroid_x - featB.centroid_x)**2 + (featA.centroid_y - featB.centroid_y)**2)

    # Boundary adjacency (distance between masks)
    maskA_u8 = (maskA.astype(np.uint8)) * 255
    maskB_u8 = (maskB.astype(np.uint8)) * 255
    dist_transform_B = cv2.distanceTransform(255 - maskB_u8, cv2.DIST_L2, 5)
    min_dist = float(np.min(dist_transform_B[maskA])) if np.sum(maskA) > 0 else 999.0

    if min_dist <= 1.0:  # Direct contact / overlap
        spatial_compat = 1.0
    else:
        spatial_compat = float(max(0.0, 1.0 - (min_dist / config.spatial_proximity_threshold_px)))

    # 3. Contamination Penalty Difference
    # If one candidate is clean foreground and the other is a border background object, penalize
    contam_diff = abs(featA.background_contamination_score - featB.background_contamination_score)
    contam_penalty = 0.5 * contam_diff if (featA.background_contamination_score > 0.40 or featB.background_contamination_score > 0.40) else 0.0

    compatibility = float(np.clip(0.50 * spatial_compat + 0.50 * depth_compat - contam_penalty, 0.0, 1.0))
    return compatibility


def generate_candidate_groups(
    candidate_masks_by_id: Dict[int, np.ndarray],
    features_list: List[CandidateFeatures],
    scores_list: List[CandidateScore],
    depth_map: np.ndarray,
    rgb_array: np.ndarray,
    config: SubjectSelectionConfig = SubjectSelectionConfig()
) -> List[CandidateGroup]:
    """
    Constructs a candidate compatibility graph and identifies coherent compound subject groups
    (e.g., Vishnu + Shesha body + Shesha heads + ornaments).
    Evaluates individual candidates as single-element groups and tests multi-element grouped candidates.
    """
    h, w = depth_map.shape
    num_cand = len(features_list)
    if num_cand == 0:
        return []

    groups: List[CandidateGroup] = []
    score_map = {s.candidate_id: s for s in scores_list}

    # 1. Add single-candidate groups
    for feat in features_list:
        cid = feat.candidate_id
        mask = candidate_masks_by_id[cid]
        groups.append(CandidateGroup(
            group_id=cid,
            candidate_ids=[cid],
            merged_mask=mask,
            combined_score=score_map[cid]
        ))

    # 2. Test pairwise and multi-candidate compound subject grouping for all plausible candidates
    top_feats = features_list[:min(16, num_cand)]

    for idx1 in range(len(top_feats)):
        for idx2 in range(idx1 + 1, len(top_feats)):
            featA = top_feats[idx1]
            featB = top_feats[idx2]
            maskA = candidate_masks_by_id[featA.candidate_id]
            maskB = candidate_masks_by_id[featB.candidate_id]

            compat = compute_pairwise_compatibility(featA, featB, maskA, maskB, config)

            if compat >= config.min_pairwise_compatibility:
                # Merge masks
                merged = merge_candidate_masks([maskA, maskB])
                iouA = compute_mask_iou(merged, maskA)
                iouB = compute_mask_iou(merged, maskB)

                # Avoid redundant merges or oversized merges (> max_subject_area_ratio)
                if iouA < 0.95 and iouB < 0.95 and (np.sum(merged) / (h * w)) <= config.max_subject_area_ratio:
                    refined = refine_subject_mask(merged, rgb_array)
                    group_id = 100 + len(groups) + 1

                    # Extract features and score for merged group
                    merged_features = extract_candidate_features(
                        refined,
                        sam_score=max(featA.sam_confidence, featB.sam_confidence),
                        prompt_origin=f"group_{featA.candidate_id}_{featB.candidate_id}",
                        depth_map=depth_map,
                        rgb_array=rgb_array,
                        candidate_id=group_id,
                        config=config
                    )
                    merged_score = score_candidate_features(merged_features, config)

                    groups.append(CandidateGroup(
                        group_id=group_id,
                        candidate_ids=[featA.candidate_id, featB.candidate_id],
                        merged_mask=refined,
                        combined_score=merged_score,
                        compatibility_matrix={(featA.candidate_id, featB.candidate_id): compat}
                    ))

    # Sort groups descending by final_score
    sorted_groups = sorted(groups, key=lambda g: g.combined_score.final_score, reverse=True)
    return sorted_groups
