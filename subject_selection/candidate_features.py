"""
Candidate Feature Extraction for Semantic Subject Selection Module.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from .schemas import CandidateFeatures, SubjectSelectionConfig


def extract_candidate_features(
    mask_bool: np.ndarray,
    sam_score: float,
    prompt_origin: str,
    depth_map: np.ndarray,
    rgb_array: np.ndarray,
    candidate_id: int,
    config: SubjectSelectionConfig = SubjectSelectionConfig()
) -> CandidateFeatures:
    """
    Computes comprehensive, deterministic geometric, spatial, border, and depth features
    for a candidate mask. Guarantees clean handling for empty or edge cases without warnings.
    """
    h, w = depth_map.shape
    total_pixels = max(1, h * w)

    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    mask_area = int(np.sum(mask_bool))
    mask_area_ratio = float(mask_area / total_pixels)

    if mask_area == 0:
        y_indices, x_indices = np.array([], dtype=int), np.array([], dtype=int)
    else:
        y_indices, x_indices = np.where(mask_bool)

    if len(y_indices) == 0:
        bbox = (0, 0, 0, 0)
        bbox_w, bbox_h = 0, 0
        bbox_area_ratio = 0.0
        centroid_x, centroid_y = 0.0, 0.0
        norm_cx, norm_cy = 0.0, 0.0
        aspect_ratio = 1.0
    else:
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
        bbox = (ymin, xmin, ymax, xmax)
        bbox_w = max(1, xmax - xmin + 1)
        bbox_h = max(1, ymax - ymin + 1)
        bbox_area_ratio = float((bbox_w * bbox_h) / total_pixels)
        centroid_y = float(np.mean(y_indices))
        centroid_x = float(np.mean(x_indices))
        norm_cy = float(centroid_y / max(1, h))
        norm_cx = float(centroid_x / max(1, w))
        aspect_ratio = float(bbox_w / bbox_h)

    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    # Background label is 0
    if num_labels > 1:
        comp_areas = stats[1:, cv2.CC_STAT_AREA]
        connected_component_count = int(num_labels - 1)
        largest_comp_area = int(np.max(comp_areas)) if len(comp_areas) > 0 else 0
        largest_component_ratio = float(largest_comp_area / max(1, mask_area))
    else:
        connected_component_count = 0
        largest_component_ratio = 0.0

    # Hole count analysis using contours
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hole_count = 0
    boundary_length = 0.0
    if hierarchy is not None and len(contours) > 0:
        # Parent-child relationship in hierarchy
        for i, h_info in enumerate(hierarchy[0]):
            boundary_length += float(cv2.arcLength(contours[i], True))
            if h_info[3] != -1:  # Has a parent -> internal hole
                hole_count += 1

    bbox_perimeter = 2.0 * (bbox_w + bbox_h)
    boundary_complexity = float(boundary_length / max(1.0, bbox_perimeter))

    # Edge alignment with RGB intensity gradients
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    rgb_edges = cv2.Canny(gray, 50, 150)
    mask_edges = cv2.Canny(mask_uint8, 50, 150)
    if np.sum(mask_edges > 0) > 0:
        aligned_pixels = np.sum((mask_edges > 0) & (rgb_edges > 0))
        edge_alignment = float(aligned_pixels / np.sum(mask_edges > 0))
    else:
        edge_alignment = 0.0

    # Depth statistics
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    depth_span = max(d_max - d_min, 1e-5)

    if mask_area > 0:
        mask_depths = depth_map[mask_bool]
        fg_depth_mean = float(np.mean(mask_depths))
        fg_depth_median = float(np.median(mask_depths))
        fg_depth_std = float(np.std(mask_depths))
    else:
        fg_depth_mean, fg_depth_median, fg_depth_std = d_max, d_max, 0.0

    # Surrounding ring depth (dilate mask by 15px - mask_bool)
    dilated_mask = cv2.dilate(mask_uint8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    surrounding_zone = dilated_mask & (~mask_bool)
    surrounding_area = np.sum(surrounding_zone)

    if surrounding_area > 0:
        surrounding_depth_mean = float(np.mean(depth_map[surrounding_zone]))
    else:
        surrounding_depth_mean = d_max

    # In rendering coordinates, smaller depth Z = closer to camera
    # Foreground should be closer than surrounding background: surrounding_depth_mean - fg_depth_mean > 0
    depth_separation = float((surrounding_depth_mean - fg_depth_mean) / depth_span)
    depth_saliency = float(1.0 - (fg_depth_mean - d_min) / depth_span)

    # Frame border contact
    border_frame = np.zeros((h, w), dtype=bool)
    border_frame[0, :] = border_frame[-1, :] = border_frame[:, 0] = border_frame[:, -1] = True
    total_border_pixels = max(1, np.sum(border_frame))
    border_touch_pixels = int(np.sum(mask_bool & border_frame))
    border_touch_ratio = float(border_touch_pixels / total_border_pixels)

    # Spatial center distance
    center_y, center_x = h / 2.0, w / 2.0
    diag_len = np.sqrt(h**2 + w**2)
    center_dist = float(np.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2) / (0.5 * diag_len))

    # Regional ratios
    lower_region_ratio = float(np.sum(mask_bool[int(h * 0.5):, :]) / max(1, mask_area))
    upper_region_ratio = float(np.sum(mask_bool[:int(h * 0.5), :]) / max(1, mask_area))
    left_region_ratio = float(np.sum(mask_bool[:, :int(w * 0.5)]) / max(1, mask_area))
    right_region_ratio = float(np.sum(mask_bool[:, int(w * 0.5):]) / max(1, mask_area))

    # Background contamination score
    # Contamination is high if mask touches frame border heavily, is far from center, has low depth separation, or extends across top corners
    border_contam = border_touch_ratio
    far_center_contam = max(0.0, center_dist - 0.5)
    far_depth_contam = max(0.0, (fg_depth_mean - (d_min + 0.6 * depth_span)) / depth_span)
    background_contamination_score = float(np.clip(
        0.40 * border_contam + 0.30 * far_center_contam + 0.30 * far_depth_contam, 0.0, 1.0
    ))

    # Fragmentation score (high component count & small largest component ratio)
    frag_count_score = min(1.0, max(0, connected_component_count - 1) / 5.0)
    frag_size_score = 1.0 - largest_component_ratio
    fragmentation_score = float(np.clip(0.5 * frag_count_score + 0.5 * frag_size_score, 0.0, 1.0))

    # Geometric coherence score
    bbox_compactness = float(mask_area / max(1, bbox_w * bbox_h))
    geometric_coherence_score = float(np.clip(0.6 * largest_component_ratio + 0.4 * bbox_compactness, 0.0, 1.0))

    # Depth coherence score (low depth standard deviation)
    norm_depth_std = fg_depth_std / depth_span
    depth_coherence_score = float(np.clip(1.0 - (norm_depth_std / 0.3), 0.0, 1.0))

    return CandidateFeatures(
        candidate_id=candidate_id,
        prompt_origin=prompt_origin,
        sam_confidence=float(sam_score),
        mask_area=mask_area,
        mask_area_ratio=mask_area_ratio,
        bbox=bbox,
        bbox_width=bbox_w,
        bbox_height=bbox_h,
        bbox_area_ratio=bbox_area_ratio,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        norm_centroid_x=norm_cx,
        norm_centroid_y=norm_cy,
        aspect_ratio=aspect_ratio,
        connected_component_count=connected_component_count,
        largest_component_ratio=largest_component_ratio,
        hole_count=hole_count,
        boundary_length=boundary_length,
        boundary_complexity=boundary_complexity,
        edge_alignment=edge_alignment,
        foreground_depth_mean=fg_depth_mean,
        foreground_depth_median=fg_depth_median,
        foreground_depth_std=fg_depth_std,
        surrounding_depth_mean=surrounding_depth_mean,
        depth_separation=depth_separation,
        depth_saliency=depth_saliency,
        border_touch_ratio=border_touch_ratio,
        image_center_distance=center_dist,
        lower_region_ratio=lower_region_ratio,
        upper_region_ratio=upper_region_ratio,
        left_region_ratio=left_region_ratio,
        right_region_ratio=right_region_ratio,
        background_contamination_score=background_contamination_score,
        fragmentation_score=fragmentation_score,
        geometric_coherence_score=geometric_coherence_score,
        depth_coherence_score=depth_coherence_score
    )


def compute_batch_relative_features(
    features_list: List[CandidateFeatures],
    candidates_masks_by_id: Dict[int, np.ndarray],
    depth_map: np.ndarray,
    config: SubjectSelectionConfig = SubjectSelectionConfig()
) -> List[CandidateFeatures]:
    """
    Computes candidate competition and relative scene features across all candidate masks:
    - relative_visual_prominence
    - foreground_cluster_distance
    - compound_subject_likelihood
    - environmental_isolation_score
    """
    if not features_list:
        return features_list

    h, w = depth_map.shape

    # Calculate weighted visual center-of-mass for top central/foreground salient candidates
    salient_feats = [f for f in features_list if f.depth_saliency > 0.40 and f.image_center_distance < 0.70]
    if not salient_feats:
        salient_feats = features_list

    cluster_weight_sum = sum(f.mask_area * f.depth_saliency for f in salient_feats)
    if cluster_weight_sum > 0:
        cluster_center_x = sum(f.centroid_x * f.mask_area * f.depth_saliency for f in salient_feats) / cluster_weight_sum
        cluster_center_y = sum(f.centroid_y * f.mask_area * f.depth_saliency for f in salient_feats) / cluster_weight_sum
    else:
        cluster_center_x, cluster_center_y = w / 2.0, h / 2.0

    diag_len = max(1.0, np.sqrt(h**2 + w**2))

    for feat in features_list:
        # 1. Relative visual prominence
        max_single_area = max((f.mask_area for f in features_list), default=1)
        feat.relative_visual_prominence = float(feat.mask_area / max(1, max_single_area))

        # 2. Foreground cluster distance
        c_dist = np.sqrt((feat.centroid_x - cluster_center_x)**2 + (feat.centroid_y - cluster_center_y)**2)
        feat.foreground_cluster_distance = float(c_dist / (0.5 * diag_len))

        # 3. Compound subject likelihood (sum of compatibility with other central/foreground candidates)
        other_compat_sum = 0.0
        maskA = candidates_masks_by_id.get(feat.candidate_id)
        if maskA is not None:
            for other in features_list:
                if other.candidate_id != feat.candidate_id and other.depth_saliency > 0.40:
                    maskB = candidates_masks_by_id.get(other.candidate_id)
                    if maskB is not None:
                        d_diff = abs(feat.foreground_depth_mean - other.foreground_depth_mean)
                        cent_dist = np.sqrt((feat.centroid_x - other.centroid_x)**2 + (feat.centroid_y - other.centroid_y)**2)
                        if cent_dist < 0.4 * diag_len and d_diff < 1.5:
                            other_compat_sum += other.mask_area_ratio
        feat.compound_subject_likelihood = float(np.clip(other_compat_sum, 0.0, 1.0))

        # 4. Environmental isolation score
        isolation = (
            0.40 * min(1.0, feat.foreground_cluster_distance) +
            0.30 * (1.0 - feat.relative_visual_prominence) +
            0.30 * (1.0 - feat.compound_subject_likelihood)
        )
        feat.environmental_isolation_score = float(np.clip(isolation, 0.0, 1.0))

    return features_list
