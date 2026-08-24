"""
Parallax Region & Spatial Depth Subsystem for First-Principles Cinematic 2.5D Renderer.

Defines the ParallaxRegion data structure and algorithms for spatial depth statistics,
depth discontinuity detection, and region classification.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2


class DepthStructureType(str, Enum):
    UNIFORM_DEPTH = "UNIFORM_DEPTH"
    GRADIENT_SURFACE = "GRADIENT_SURFACE"
    MULTI_DEPTH_OBJECT = "MULTI_DEPTH_OBJECT"
    DEPTH_DISCONTINUITY = "DEPTH_DISCONTINUITY"
    UNCERTAIN_DEPTH = "UNCERTAIN_DEPTH"


class EdgeAlignmentStatus(str, Enum):
    ALIGNED = "ALIGNED"
    PARTIALLY_ALIGNED = "PARTIALLY_ALIGNED"
    MISALIGNED = "MISALIGNED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ParallaxRegion:
    """
    Spatially and semantically coherent region for 2.5D view synthesis.
    Contains detailed continuous depth statistics, boundary attributes, and rigidity estimates.
    """
    region_id: str
    entity_ids: List[str]
    semantic_role: str
    parent_region_id: Optional[str]
    mask: np.ndarray
    area: int
    centroid: Tuple[float, float]
    depth_mean: float
    depth_median: float
    depth_p10: float
    depth_p25: float
    depth_p50: float
    depth_p75: float
    depth_p90: float
    depth_std: float
    depth_iqr: float
    local_depth_gradient: float
    depth_discontinuity_score: float
    boundary_strength: float
    occlusion_boundary_mask: np.ndarray
    disocclusion_risk: float
    rigidity_score: float
    attachment_score: float
    support_score: float
    independent_motion_allowed: bool
    motion_coupling_group: str
    motion_eligibility: str
    parallax_priority: int
    confidence: float
    depth_structure_type: DepthStructureType

    def to_dict(self) -> Dict[str, Any]:
        """Convert region metadata to serializable dictionary (omitting heavy mask arrays)."""
        y_idx, x_idx = np.where(self.mask)
        bbox = [
            int(np.min(y_idx)) if len(y_idx) > 0 else 0,
            int(np.min(x_idx)) if len(x_idx) > 0 else 0,
            int(np.max(y_idx)) if len(y_idx) > 0 else 0,
            int(np.max(x_idx)) if len(x_idx) > 0 else 0,
        ]
        return {
            "region_id": self.region_id,
            "entity_ids": self.entity_ids,
            "semantic_role": self.semantic_role,
            "parent_region_id": self.parent_region_id,
            "area": self.area,
            "bounding_box": bbox,
            "centroid": [round(float(self.centroid[0]), 2), round(float(self.centroid[1]), 2)],
            "depth_mean": round(float(self.depth_mean), 4),
            "depth_median": round(float(self.depth_median), 4),
            "depth_p10": round(float(self.depth_p10), 4),
            "depth_p25": round(float(self.depth_p25), 4),
            "depth_p50": round(float(self.depth_p50), 4),
            "depth_p75": round(float(self.depth_p75), 4),
            "depth_p90": round(float(self.depth_p90), 4),
            "depth_std": round(float(self.depth_std), 4),
            "depth_iqr": round(float(self.depth_iqr), 4),
            "local_depth_gradient": round(float(self.local_depth_gradient), 4),
            "depth_discontinuity_score": round(float(self.depth_discontinuity_score), 4),
            "boundary_strength": round(float(self.boundary_strength), 4),
            "disocclusion_risk": round(float(self.disocclusion_risk), 4),
            "rigidity_score": round(float(self.rigidity_score), 4),
            "attachment_score": round(float(self.attachment_score), 4),
            "support_score": round(float(self.support_score), 4),
            "independent_motion_allowed": self.independent_motion_allowed,
            "motion_coupling_group": self.motion_coupling_group,
            "motion_eligibility": self.motion_eligibility,
            "parallax_priority": self.parallax_priority,
            "confidence": round(float(self.confidence), 4),
            "depth_structure_type": self.depth_structure_type.value,
        }


def compute_region_depth_statistics(
    mask: np.ndarray,
    depth_map: np.ndarray
) -> Dict[str, float]:
    """
    Computes spatial depth distribution statistics across a masked region.
    """
    if not np.any(mask):
        return {
            "mean": 5.0, "median": 5.0, "p10": 5.0, "p25": 5.0,
            "p50": 5.0, "p75": 5.0, "p90": 5.0, "std": 0.0, "iqr": 0.0
        }

    depths = depth_map[mask]
    q10, q25, q50, q75, q90 = np.quantile(depths, [0.10, 0.25, 0.50, 0.75, 0.90])

    return {
        "mean": float(np.mean(depths)),
        "median": float(q50),
        "p10": float(q10),
        "p25": float(q25),
        "p50": float(q50),
        "p75": float(q75),
        "p90": float(q90),
        "std": float(np.std(depths)),
        "iqr": float(q75 - q25)
    }


def classify_depth_rgb_edge_alignment(
    depth_map: np.ndarray,
    rgb_array: np.ndarray
) -> Tuple[EdgeAlignmentStatus, float, np.ndarray]:
    """
    Classifies alignment between depth gradient edges and RGB color boundaries.
    Suppresses unaligned depth edges to eliminate depth-edge halos and double edges.
    Returns: (alignment_status, alignment_iou, validated_depth_edge_mask)
    """
    gx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    thresh = float(np.percentile(grad_mag, 90.0))
    d_edges = grad_mag > max(0.5, thresh)

    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    c_edges = cv2.Canny(gray, 50, 150) > 0

    # Dilate Canny edges by 3px to allow minor alignment tolerance
    c_dilated = cv2.dilate(c_edges.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0

    # Validated depth edges are depth edges that lie within dilated RGB boundaries
    validated_d_edges = d_edges & c_dilated

    intersection = np.sum(validated_d_edges)
    union = np.sum(d_edges | c_edges)
    iou = float(intersection / max(1, union))

    if iou >= 0.50:
        status = EdgeAlignmentStatus.ALIGNED
    elif iou >= 0.25:
        status = EdgeAlignmentStatus.PARTIALLY_ALIGNED
    elif iou >= 0.10:
        status = EdgeAlignmentStatus.UNCERTAIN
    else:
        status = EdgeAlignmentStatus.MISALIGNED

    return status, iou, validated_d_edges


def detect_depth_discontinuities(
    depth_map: np.ndarray,
    rgb_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes continuous depth gradient magnitude |∇depth| fused with RGB edge alignment validation.
    Returns: (depth_gradient_map, validated_depth_edge_mask)
    """
    gx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    status, iou, validated_edge_mask = classify_depth_rgb_edge_alignment(depth_map, rgb_array)

    return grad_mag, validated_edge_mask


def construct_parallax_regions(
    entities: List[Any],
    depth_map: np.ndarray,
    rgb_array: np.ndarray
) -> List[ParallaxRegion]:
    """
    Constructs ParallaxRegion objects for consolidated entities and depth fields.
    """
    grad_mag, depth_edges = detect_depth_discontinuities(depth_map, rgb_array)
    regions = []

    for idx, entity in enumerate(entities):
        mask = entity.mask
        if not np.any(mask):
            continue

        stats = compute_region_depth_statistics(mask, depth_map)
        y_idx, x_idx = np.where(mask)
        cy, cx = float(np.mean(y_idx)), float(np.mean(x_idx))

        # Classify depth structure
        if stats["std"] < 0.2:
            struct_type = DepthStructureType.UNIFORM_DEPTH
        elif stats["iqr"] > 1.5:
            struct_type = DepthStructureType.MULTI_DEPTH_OBJECT
        elif float(np.mean(grad_mag[mask])) > 0.8:
            struct_type = DepthStructureType.DEPTH_DISCONTINUITY
        else:
            struct_type = DepthStructureType.GRADIENT_SURFACE

        is_sub = entity.semantic_role == "PRIMARY_SUBJECT"
        reg = ParallaxRegion(
            region_id=f"REG_{idx:02d}_{entity.name.upper()}",
            entity_ids=[entity.entity_id],
            semantic_role=entity.semantic_role,
            parent_region_id=None,
            mask=mask,
            area=int(np.sum(mask)),
            centroid=(cx, cy),
            depth_mean=stats["mean"],
            depth_median=stats["median"],
            depth_p10=stats["p10"],
            depth_p25=stats["p25"],
            depth_p50=stats["p50"],
            depth_p75=stats["p75"],
            depth_p90=stats["p90"],
            depth_std=stats["std"],
            depth_iqr=stats["iqr"],
            local_depth_gradient=float(np.mean(grad_mag[mask])),
            depth_discontinuity_score=float(np.mean(depth_edges[mask])),
            boundary_strength=0.85,
            occlusion_boundary_mask=depth_edges & mask,
            disocclusion_risk=0.15 if is_sub else 0.05,
            rigidity_score=0.95 if is_sub else 0.80,
            attachment_score=1.0 if is_sub else 0.5,
            support_score=1.0 if is_sub else 0.5,
            independent_motion_allowed=not is_sub,
            motion_coupling_group="PRIMARY_SUBJECT_GROUP" if is_sub else f"ENV_GROUP_{idx:02d}",
            motion_eligibility="PRIMARY_CAMERA_PARALLAX" if is_sub else "BACKGROUND_PARALLAX",
            parallax_priority=1 if is_sub else 3,
            confidence=float(getattr(entity, "trust_score", 0.90)),
            depth_structure_type=struct_type
        )
        regions.append(reg)

    return regions


@dataclass
class OcclusionBoundary:
    """
    Explicit occlusion boundary between foreground and background parallax regions.
    Tracks depth delta, confidence, and expected disocclusion exposure width.
    """
    boundary_id: str
    foreground_region_id: str
    background_region_id: str
    boundary_mask: np.ndarray
    depth_delta: float
    confidence: float
    exposure_direction: Tuple[float, float]
    expected_disocclusion_width_px: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "foreground_region_id": self.foreground_region_id,
            "background_region_id": self.background_region_id,
            "boundary_pixel_count": int(np.sum(self.boundary_mask)),
            "depth_delta": round(float(self.depth_delta), 4),
            "confidence": round(float(self.confidence), 4),
            "exposure_direction": [round(float(self.exposure_direction[0]), 2), round(float(self.exposure_direction[1]), 2)],
            "expected_disocclusion_width_px": round(float(self.expected_disocclusion_width_px), 2)
        }


def forecast_disocclusion_regions(
    regions: List[ParallaxRegion],
    camera_translation: np.ndarray,
    fx: float
) -> Dict[str, Any]:
    """
    Forecasts newly exposed background pixels and exposure area for a candidate camera translation.
    """
    tx, ty, tz = camera_translation[0], camera_translation[1], camera_translation[2]
    total_forecast_area = 0
    region_forecasts = []

    for reg in regions:
        if reg.semantic_role == "PRIMARY_SUBJECT":
            # Estimate boundary disparity shift delta_u = (fx * tx) / Z
            disp_shift = abs((fx * tx) / max(1.0, reg.depth_mean))
            boundary_pixels = int(np.sum(reg.occlusion_boundary_mask))
            exposed_area = int(boundary_pixels * min(disp_shift, 10.0))
            total_forecast_area += exposed_area

            region_forecasts.append({
                "region_id": reg.region_id,
                "expected_disocclusion_px": round(float(disp_shift), 2),
                "forecast_area_px": exposed_area,
                "reconstruction_confidence": 0.85
            })

    return {
        "total_forecast_area_px": total_forecast_area,
        "camera_translation": [round(float(c), 4) for c in camera_translation],
        "region_forecasts": region_forecasts
    }
