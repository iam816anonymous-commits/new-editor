"""
Subject Confidence Model for Semantic Subject Selection Module.
"""

import numpy as np
from .schemas import CandidateFeatures, CandidateScore, SubjectConfidence, SubjectSelectionConfig


def compute_subject_confidence(
    features: CandidateFeatures,
    score: CandidateScore,
    config: SubjectSelectionConfig = SubjectSelectionConfig()
) -> SubjectConfidence:
    """
    Computes an explicit, multi-layer SubjectConfidence model.
    CRITICAL ARCHITECTURAL RULE: SAM prediction confidence is NOT subject confidence.
    This module measures true semantic subject confidence by combining model confidence,
    geometric coherence, depth consistency, and composition.
    """
    # 1. SAM model prediction confidence
    sam_model_conf = float(np.clip(features.sam_confidence, 0.0, 1.0))

    # 2. Geometric confidence (large component, reasonable aspect ratio, low fragmentation)
    geom_conf = float(np.clip(
        0.50 * features.geometric_coherence_score +
        0.50 * (1.0 - features.fragmentation_score),
        0.0, 1.0
    ))

    # 3. Depth confidence (low depth std inside subject, strong separation from surrounding background)
    depth_conf = float(np.clip(
        0.50 * features.depth_coherence_score +
        0.50 * min(1.0, max(0.0, features.depth_separation + 0.5)),
        0.0, 1.0
    ))

    # 4. Semantic subject confidence (combination of centrality, depth saliency, relative prominence, and low environmental isolation)
    semantic_conf = float(np.clip(
        0.30 * score.centrality_score +
        0.30 * score.depth_saliency_score +
        0.20 * features.relative_visual_prominence +
        0.20 * (1.0 - features.environmental_isolation_score),
        0.0, 1.0
    ))

    # 5. Final Subject Confidence
    final_conf = float(np.clip(
        0.15 * sam_model_conf +
        0.20 * geom_conf +
        0.20 * depth_conf +
        0.45 * semantic_conf -
        0.35 * features.environmental_isolation_score,
        0.0, 1.0
    ))

    return SubjectConfidence(
        sam_model_confidence=round(sam_model_conf, 4),
        geometric_confidence=round(geom_conf, 4),
        depth_confidence=round(depth_conf, 4),
        semantic_subject_confidence=round(semantic_conf, 4),
        final_subject_confidence=round(final_conf, 4)
    )
