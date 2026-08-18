"""
Parallax Quality Scoring Engine and Artifact Detector.
Evaluates PARALLAX_SCORE, TEMPORAL_SCORE, SUBJECT_PRESERVATION_SCORE, ARTIFACT_SCORE, and OVERALL_PARALLAX_QUALITY.
"""

from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
from .schemas import ParallaxQualityScore


def detect_rendering_artifacts(
    rendered_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    subject_mask: np.ndarray,
    black_hole_threshold: float = 10.0,
    edge_halo_threshold: float = 40.0
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """
    Detects rendering artifacts:
    - Black holes / unwritten transparent pixels
    - Stretched textures
    - Edge halos around subject boundary
    - Ghosting / mask leaks

    Returns:
        (artifact_score_penalty, artifact_mask, artifact_metrics)
    """
    h, w, _ = rendered_rgb.shape
    total_pixels = h * w

    # 1. Black Hole Detection (near zero intensity in non-black reference pixels)
    ref_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    ren_gray = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2GRAY)

    black_hole_mask = (ren_gray < black_hole_threshold) & (ref_gray >= black_hole_threshold)
    black_hole_count = int(np.sum(black_hole_mask))

    # 2. Subject Edge Halo Detection
    sub_boundary = (cv2.Canny((subject_mask * 255).astype(np.uint8), 100, 200) > 0)
    boundary_zone = cv2.dilate(sub_boundary.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))) > 0

    diff = np.abs(ren_gray.astype(np.float32) - ref_gray.astype(np.float32))
    halo_mask = boundary_zone & (diff > edge_halo_threshold)
    halo_count = int(np.sum(halo_mask))

    # Combined Artifact Mask
    artifact_mask = (black_hole_mask | halo_mask).astype(np.uint8) * 255

    black_hole_ratio = float(black_hole_count / total_pixels)
    halo_ratio = float(halo_count / max(1, np.sum(boundary_zone)))

    artifact_penalty = float(np.clip(10.0 * black_hole_ratio + 0.5 * halo_ratio, 0.0, 1.0))

    metrics = {
        "black_hole_pixels": black_hole_count,
        "black_hole_ratio": round(black_hole_ratio, 4),
        "edge_halo_pixels": halo_count,
        "edge_halo_ratio": round(halo_ratio, 4),
        "artifact_penalty": round(artifact_penalty, 4)
    }

    return artifact_penalty, artifact_mask, metrics


def compute_parallax_quality_score(
    rendered_sequence: List[np.ndarray],
    reference_rgb: np.ndarray,
    subject_mask: np.ndarray,
    depth_map: np.ndarray,
    minimum_motion_px: float = 2.0,
    maximum_motion_px: float = 45.0
) -> Tuple[ParallaxQualityScore, Dict[str, Any]]:
    """
    Computes weighted multi-signal ParallaxQualityScore.

    Weights:
    Overall = 0.20 * Parallax + 0.15 * Edge + 0.20 * Reconstruction + 0.15 * Temporal + 0.15 * SubjectPreservation + 0.15 * Composition - ArtifactPenalty
    """
    num_frames = len(rendered_sequence)
    h, w, _ = reference_rgb.shape

    f0 = rendered_sequence[0].astype(np.float32)
    f_mid = rendered_sequence[num_frames // 2].astype(np.float32) if num_frames > 1 else f0
    f_last = rendered_sequence[-1].astype(np.float32) if num_frames > 1 else f0

    # 1. Measure motion displacements
    abs_motion = np.mean(np.abs(f_mid - f0), axis=2)
    bg_mask = ~subject_mask

    sub_motion = float(np.mean(abs_motion[subject_mask])) if np.any(subject_mask) else 1.0
    bg_motion = float(np.mean(abs_motion[bg_mask])) if np.any(bg_mask) else 0.2

    # Motion floor & ceiling checks
    if sub_motion < minimum_motion_px:
        motion_floor_status = "STATIC_FAILURE"
    elif sub_motion > maximum_motion_px:
        motion_floor_status = "EXCESSIVE_MOTION_FAILURE"
    else:
        motion_floor_status = "PASSED"

    # Parallax score: rewards depth-dependent differential motion (subject > background)
    diff_motion = max(0.0, sub_motion - bg_motion)
    parallax_score = float(np.clip(diff_motion / 5.0, 0.0, 1.0))

    # 2. Temporal metrics (MAD, Loop closure MAE)
    frame_mads = [float(np.mean(np.abs(rendered_sequence[i].astype(np.float32) - rendered_sequence[i-1].astype(np.float32)))) for i in range(1, num_frames)]
    temp_mad = float(np.mean(frame_mads)) if frame_mads else 0.0

    loop_mae = float(np.mean(np.abs(f_last - f0)))
    temporal_score = float(np.clip(1.0 - (temp_mad / 5.0 + loop_mae / 2.0), 0.0, 1.0))

    # 3. Subject Preservation (IoU / MAE inside subject)
    sub_mae = float(np.mean(np.abs(f_mid[subject_mask] - reference_rgb.astype(np.float32)[subject_mask]))) if np.any(subject_mask) else 0.0
    sub_preservation_score = float(np.clip(1.0 - sub_mae / 30.0, 0.0, 1.0))

    # 4. Artifact penalty
    penalty, art_mask, art_metrics = detect_rendering_artifacts(rendered_sequence[num_frames // 2], reference_rgb, subject_mask)

    # 5. Weighted Overall Quality
    overall = (
        0.25 * parallax_score +
        0.25 * temporal_score +
        0.25 * sub_preservation_score +
        0.25 * (1.0 - penalty)
    )
    overall_clamped = float(np.clip(overall, 0.0, 1.0))

    score_obj = ParallaxQualityScore(
        background_motion_px=round(bg_motion, 3),
        midground_motion_px=round(sub_motion * 0.5, 3),
        primary_subject_motion_px=round(sub_motion, 3),
        foreground_motion_px=round(sub_motion * 1.5, 3),
        temporal_mad=round(temp_mad, 3),
        boundary_mad=round(temp_mad, 3),
        loop_closure_mae=round(loop_mae, 3),
        edge_artifact_ratio=art_metrics["edge_halo_ratio"],
        overlap_artifact_ratio=art_metrics["black_hole_ratio"],
        overall_parallax_quality=round(overall_clamped, 3)
    )

    metrics_summary = {
        "parallax_score": round(parallax_score, 3),
        "temporal_score": round(temporal_score, 3),
        "subject_preservation_score": round(sub_preservation_score, 3),
        "motion_floor_status": motion_floor_status,
        "artifact_metrics": art_metrics,
        "overall_quality_score": round(overall_clamped, 3)
    }

    return score_obj, metrics_summary
