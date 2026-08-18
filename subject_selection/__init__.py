"""
Semantic Subject Selection Sub-Module for First-Principles Cinematic 2.5D Parallax Renderer (V0).
"""

from .schemas import (
    SubjectSelectionConfig,
    CandidateFeatures,
    CandidateScore,
    CandidateGroup,
    SubjectConfidence,
    MaskValidationResult,
    SubjectSelectionResult
)
from .subject_selector import select_semantic_subject

__all__ = [
    "SubjectSelectionConfig",
    "CandidateFeatures",
    "CandidateScore",
    "CandidateGroup",
    "SubjectConfidence",
    "MaskValidationResult",
    "SubjectSelectionResult",
    "select_semantic_subject"
]
