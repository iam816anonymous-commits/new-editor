"""
Typed Dataclasses and Configuration Schemas for Semantic Subject Selection Module.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


@dataclass
class SubjectSelectionConfig:
    """Configurable weights, penalties, and thresholds for semantic subject selection."""
    # Score component weights
    weight_sam_confidence: float = 0.10
    weight_centrality: float = 0.20
    weight_depth_saliency: float = 0.20
    weight_scale: float = 0.15
    weight_relative_prominence: float = 0.20
    weight_compound_support: float = 0.15
    weight_geometric_coherence: float = 0.05
    weight_edge_alignment: float = 0.05

    # Penalties
    weight_border_penalty: float = 0.35
    weight_contamination_penalty: float = 0.35
    weight_fragmentation_penalty: float = 0.20
    weight_environmental_penalty: float = 0.35

    # Target area coverage ratios
    min_subject_area_ratio: float = 0.01   # 1% minimum coverage
    optimal_area_ratio_min: float = 0.08   # 8% optimal coverage lower bound
    optimal_area_ratio_max: float = 0.55   # 55% optimal coverage upper bound
    max_subject_area_ratio: float = 0.80   # 80% maximum coverage

    # Grouping thresholds
    spatial_proximity_threshold_px: float = 40.0
    depth_similarity_threshold: float = 1.2
    min_pairwise_compatibility: float = 0.55

    # Validation & Acceptance gate thresholds
    min_final_subject_confidence: float = 0.45
    min_score_margin: float = 0.02
    max_allowed_background_contamination: float = 0.35
    max_allowed_border_touch_ratio: float = 0.25


@dataclass
class CandidateFeatures:
    """Structured, deterministic geometric and depth features for a single SAM candidate mask."""
    candidate_id: int
    prompt_origin: str
    sam_confidence: float

    mask_area: int
    mask_area_ratio: float

    bbox: Tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax)
    bbox_width: int
    bbox_height: int
    bbox_area_ratio: float

    centroid_x: float
    centroid_y: float
    norm_centroid_x: float
    norm_centroid_y: float

    aspect_ratio: float
    connected_component_count: int
    largest_component_ratio: float
    hole_count: int

    boundary_length: float
    boundary_complexity: float
    edge_alignment: float

    foreground_depth_mean: float
    foreground_depth_median: float
    foreground_depth_std: float
    surrounding_depth_mean: float
    depth_separation: float
    depth_saliency: float

    border_touch_ratio: float
    image_center_distance: float

    lower_region_ratio: float
    upper_region_ratio: float
    left_region_ratio: float
    right_region_ratio: float

    background_contamination_score: float
    fragmentation_score: float
    geometric_coherence_score: float
    depth_coherence_score: float

    relative_visual_prominence: float = 0.0
    foreground_cluster_distance: float = 0.0
    compound_subject_likelihood: float = 0.0
    environmental_isolation_score: float = 0.0


@dataclass
class CandidateScore:
    """Detailed score breakdown for a candidate mask or candidate group."""
    candidate_id: int
    sam_confidence: float
    centrality_score: float
    depth_saliency_score: float
    scale_score: float
    relative_prominence_score: float
    compound_support_score: float
    geometric_coherence_score: float
    edge_alignment_score: float

    border_penalty: float
    contamination_penalty: float
    fragmentation_penalty: float
    environmental_penalty: float

    raw_score: float
    final_score: float


@dataclass
class CandidateGroup:
    """A compound candidate group combining multiple spatially/depth-compatible foreground masks."""
    group_id: int
    candidate_ids: List[int]
    merged_mask: np.ndarray = field(repr=False)
    combined_score: CandidateScore
    compatibility_matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)


@dataclass
class SubjectConfidence:
    """Explicit confidence model distinguishing SAM prediction score from true semantic subject confidence."""
    sam_model_confidence: float
    geometric_confidence: float
    depth_confidence: float
    semantic_subject_confidence: float
    final_subject_confidence: float


@dataclass
class MaskValidationResult:
    """Result of the mask acceptance gate."""
    validation_status: str  # "ACCEPTED", "REJECTED", "UNCERTAIN"
    is_valid: bool
    status_reasons: List[str]
    confidence: SubjectConfidence


@dataclass
class SubjectSelectionResult:
    """Complete structured output contract for semantic subject selection."""
    selected_mask: np.ndarray = field(repr=False)
    refined_mask: np.ndarray = field(repr=False)
    selected_group_ids: List[int]
    candidate_features_list: List[CandidateFeatures] = field(default_factory=list)
    candidate_scores_list: List[CandidateScore] = field(default_factory=list)
    candidate_groups: List[CandidateGroup] = field(default_factory=list)
    validation_result: MaskValidationResult = field(default_factory=lambda: MaskValidationResult("REJECTED", False, [], SubjectConfidence(0,0,0,0,0)))
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
