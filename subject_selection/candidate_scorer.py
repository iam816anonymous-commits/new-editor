"""
Multi-Signal Candidate Scoring and Ranking for Semantic Subject Selection Module.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from .schemas import CandidateFeatures, CandidateScore, SubjectSelectionConfig


def score_candidate_features(
    features: CandidateFeatures,
    config: SubjectSelectionConfig = SubjectSelectionConfig()
) -> CandidateScore:
    """
    Computes a weighted multi-signal score for a candidate mask using explicit configuration.
    Combines SAM prediction confidence, spatial centrality, depth saliency, scale coverage,
    geometric coherence, and edge alignment minus border, contamination, and fragmentation penalties.
    """
    # 1. SAM prediction confidence (0..1)
    sam_conf = float(np.clip(features.sam_confidence, 0.0, 1.0))

    # 2. Centrality score (0..1)
    centrality_score = float(np.clip(1.0 - features.image_center_distance, 0.0, 1.0))

    # 3. Depth saliency score (0..1)
    depth_saliency_score = float(np.clip(features.depth_saliency, 0.0, 1.0))

    # 4. Scale score (prefer optimal area coverage 8% .. 55%)
    cov = features.mask_area_ratio
    if cov < config.min_subject_area_ratio:
        scale_score = 0.0
    elif cov > config.max_subject_area_ratio:
        scale_score = 0.0
    elif config.optimal_area_ratio_min <= cov <= config.optimal_area_ratio_max:
        scale_score = 1.0
    elif cov < config.optimal_area_ratio_min:
        scale_score = cov / config.optimal_area_ratio_min
    else:
        scale_score = (config.max_subject_area_ratio - cov) / (config.max_subject_area_ratio - config.optimal_area_ratio_max)
    scale_score = float(np.clip(scale_score, 0.0, 1.0))

    # 5. Geometric coherence score (0..1)
    geometric_coherence_score = float(np.clip(features.geometric_coherence_score, 0.0, 1.0))

    # 6. Edge alignment score (0..1)
    edge_alignment_score = float(np.clip(features.edge_alignment, 0.0, 1.0))

    # Weighted Positive Raw Score
    raw_score = (
        config.weight_sam_confidence * sam_conf +
        config.weight_centrality * centrality_score +
        config.weight_depth_saliency * depth_saliency_score +
        config.weight_scale * scale_score +
        config.weight_geometric_coherence * geometric_coherence_score +
        config.weight_edge_alignment * edge_alignment_score
    )

    # 7. Penalties
    # Border Penalty
    b_touch = features.border_touch_ratio
    if b_touch > 0.05:
        border_penalty = float(config.weight_border_penalty * min(1.0, b_touch / config.max_allowed_border_touch_ratio))
    else:
        border_penalty = 0.0

    # Background Contamination Penalty
    contamination_penalty = float(config.weight_contamination_penalty * features.background_contamination_score)

    # Fragmentation Penalty
    fragmentation_penalty = float(config.weight_fragmentation_penalty * features.fragmentation_score)

    total_penalties = border_penalty + contamination_penalty + fragmentation_penalty
    final_score = float(max(0.0, raw_score - total_penalties))

    return CandidateScore(
        candidate_id=features.candidate_id,
        sam_confidence=round(sam_conf, 4),
        centrality_score=round(centrality_score, 4),
        depth_saliency_score=round(depth_saliency_score, 4),
        scale_score=round(scale_score, 4),
        geometric_coherence_score=round(geometric_coherence_score, 4),
        edge_alignment_score=round(edge_alignment_score, 4),
        border_penalty=round(border_penalty, 4),
        contamination_penalty=round(contamination_penalty, 4),
        fragmentation_penalty=round(fragmentation_penalty, 4),
        raw_score=round(raw_score, 4),
        final_score=round(final_score, 4)
    )


def rank_candidates(
    features_list: List[CandidateFeatures],
    scores_list: List[CandidateScore]
) -> Tuple[List[CandidateFeatures], List[CandidateScore]]:
    """Sorts candidate features and scores lists in descending order of final_score."""
    score_map = {s.candidate_id: s for s in scores_list}
    sorted_features = sorted(features_list, key=lambda f: score_map[f.candidate_id].final_score, reverse=True)
    sorted_scores = [score_map[f.candidate_id] for f in sorted_features]
    return sorted_features, sorted_scores
