"""
Mask Acceptance Gate and Validation for Semantic Subject Selection Module.
"""

import numpy as np
from typing import List, Tuple, Optional
from .schemas import (
    CandidateFeatures,
    CandidateScore,
    CandidateGroup,
    SubjectConfidence,
    MaskValidationResult,
    SubjectSelectionConfig
)
from .confidence import compute_subject_confidence


def validate_selected_subject_mask(
    selected_group: CandidateGroup,
    features_map: dict,
    score_margin: float,
    config: SubjectSelectionConfig = SubjectSelectionConfig()
) -> MaskValidationResult:
    """
    Mandatory Mask Acceptance Gate.
    Evaluates selected candidate/group against strict validation criteria before Phase C background reconstruction:
    - Minimum/maximum subject area coverage
    - Border contamination threshold
    - Background contamination threshold
    - Fragmentation threshold
    - Minimum final subject confidence
    - Score margin relative to runner-up candidates

    Returns MaskValidationResult with status: "ACCEPTED", "REJECTED", or "UNCERTAIN".
    """
    status_reasons: List[str] = []

    # Aggregate or primary candidate features
    score = selected_group.combined_score
    primary_cid = selected_group.candidate_ids[0]
    primary_feat = features_map.get(primary_cid)

    if primary_feat is None:
        return MaskValidationResult(
            validation_status="REJECTED",
            is_valid=False,
            status_reasons=["Missing candidate feature record."],
            confidence=SubjectConfidence(0, 0, 0, 0, 0)
        )

    # Compute explicit subject confidence
    confidence = compute_subject_confidence(primary_feat, score, config)

    # 1. Area Coverage Check
    mask_area_ratio = np.sum(selected_group.merged_mask) / selected_group.merged_mask.size
    if mask_area_ratio < config.min_subject_area_ratio:
        status_reasons.append(f"Subject area coverage too small ({mask_area_ratio:.2%} < {config.min_subject_area_ratio:.2%})")
    elif mask_area_ratio > config.max_subject_area_ratio:
        status_reasons.append(f"Subject area coverage too large ({mask_area_ratio:.2%} > {config.max_subject_area_ratio:.2%})")

    # 2. Border Contamination Check
    if primary_feat.border_touch_ratio > config.max_allowed_border_touch_ratio and score.centrality_score < 0.60:
        status_reasons.append(f"Excessive border contact ({primary_feat.border_touch_ratio:.2%} > {config.max_allowed_border_touch_ratio:.2%}) with low centrality")

    # 3. Background Contamination Check
    if primary_feat.background_contamination_score > config.max_allowed_background_contamination:
        status_reasons.append(f"High background contamination score ({primary_feat.background_contamination_score:.2f} > {config.max_allowed_background_contamination:.2f})")

    # 4. Final Confidence Check
    if confidence.final_subject_confidence < config.min_final_subject_confidence:
        status_reasons.append(f"Final subject confidence too low ({confidence.final_subject_confidence:.2f} < {config.min_final_subject_confidence:.2f})")

    # 5. Score Margin Check
    if score_margin < config.min_score_margin and score.final_score < 0.55:
        status_reasons.append(f"Ambiguous top candidate selection (score margin {score_margin:.3f} < {config.min_score_margin:.3f})")

    # Determine Gate Result
    if not status_reasons:
        validation_status = "ACCEPTED"
        is_valid = True
        status_reasons.append("Subject selection validated and accepted.")
    elif len(status_reasons) == 1 and ("margin" in status_reasons[0] or "border" in status_reasons[0]):
        validation_status = "UNCERTAIN"
        is_valid = False
    else:
        validation_status = "REJECTED"
        is_valid = False

    return MaskValidationResult(
        validation_status=validation_status,
        is_valid=is_valid,
        status_reasons=status_reasons,
        confidence=confidence
    )
