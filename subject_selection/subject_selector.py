"""
Main Orchestrator for Semantic Subject Selection Module.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .schemas import (
    SubjectSelectionConfig,
    CandidateFeatures,
    CandidateScore,
    CandidateGroup,
    SubjectConfidence,
    MaskValidationResult,
    SubjectSelectionResult
)
from .candidate_features import extract_candidate_features, compute_batch_relative_features
from .candidate_scorer import score_candidate_features, rank_candidates
from .candidate_grouper import generate_candidate_groups
from .mask_merger import refine_subject_mask
from .mask_validator import validate_selected_subject_mask


def generate_candidate_masks_sam2(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    predictor: SAM2ImagePredictor,
    iou_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """Generates a deduplicated list of candidate masks using SAM 2 with grid, depth point, and box prompts."""
    predictor.set_image(rgb_array)
    h, w = depth_map.shape
    total_pixels = max(1, h * w)

    prompts = []

    # 1. 5x5 Grid Point Prompts
    grid_y = np.linspace(h * 0.2, h * 0.8, 5, dtype=int)
    grid_x = np.linspace(w * 0.2, w * 0.8, 5, dtype=int)
    for py in grid_y:
        for px in grid_x:
            prompts.append(("point", np.array([[px, py]], dtype=np.float32), np.array([1], dtype=np.int32), f"grid_{px}_{py}"))

    # 2. Foreground Depth Saliency Point Prompts (5 points with smallest depth Z)
    flat_indices = np.argpartition(depth_map.ravel(), min(100, total_pixels - 1))[:min(100, total_pixels)]
    unravel_y, unravel_x = np.unravel_index(flat_indices, (h, w))
    for k in range(0, len(unravel_y), max(1, len(unravel_y) // 5))[:5]:
        fy, fx = int(unravel_y[k]), int(unravel_x[k])
        prompts.append(("point", np.array([[fx, fy]], dtype=np.float32), np.array([1], dtype=np.int32), f"fg_depth_{fx}_{fy}"))

    # 3. Center Box Prompts
    boxes = [
        (np.array([int(w * 0.20), int(h * 0.20), int(w * 0.80), int(h * 0.80)], dtype=np.float32), "box_center_20_80"),
        (np.array([int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)], dtype=np.float32), "box_central_25_75"),
        (np.array([int(w * 0.10), int(h * 0.15), int(w * 0.90), int(h * 0.85)], dtype=np.float32), "box_wide_10_90")
    ]

    raw_masks: List[Tuple[np.ndarray, float, str]] = []

    for ptype, pt_coords, pt_labels, origin in prompts:
        try:
            masks, scores, _ = predictor.predict(
                point_coords=pt_coords,
                point_labels=pt_labels,
                multimask_output=True
            )
            for mask, score in zip(masks, scores):
                mask_bool = mask.astype(bool)
                cov = np.sum(mask_bool) / total_pixels
                if 0.005 <= cov <= 0.85:
                    raw_masks.append((mask_bool, float(score), origin))
        except Exception:
            continue

    for box, origin in boxes:
        try:
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
            for mask, score in zip(masks, scores):
                mask_bool = mask.astype(bool)
                cov = np.sum(mask_bool) / total_pixels
                if 0.005 <= cov <= 0.85:
                    raw_masks.append((mask_bool, float(score), origin))
        except Exception:
            continue

    # Deduplicate candidate masks using IoU > iou_threshold
    dedup_masks: List[Dict[str, Any]] = []
    for mask_bool, score, origin in raw_masks:
        is_dup = False
        for existing in dedup_masks:
            intersection = np.sum(mask_bool & existing["mask_bool"])
            union = np.sum(mask_bool | existing["mask_bool"])
            iou = intersection / union if union > 0 else 0.0
            if iou >= iou_threshold:
                is_dup = True
                if score > existing["sam_score"]:
                    existing["mask_bool"] = mask_bool
                    existing["sam_score"] = score
                    existing["prompt_origin"] = origin
                break
        if not is_dup:
            dedup_masks.append({
                "mask_bool": mask_bool,
                "sam_score": score,
                "prompt_origin": origin
            })

    return dedup_masks


def export_diagnostics(
    hash_dir: Path,
    original_rgb: np.ndarray,
    depth_map: np.ndarray,
    features_list: List[CandidateFeatures],
    scores_list: List[CandidateScore],
    groups: List[CandidateGroup],
    selected_group: CandidateGroup,
    refined_mask: np.ndarray,
    validation_result: MaskValidationResult,
    mask_by_id: Optional[Dict[int, np.ndarray]] = None
) -> None:
    """Generates and exports diagnostic PNG artifacts and 12_validation_report.json."""
    hash_dir.mkdir(parents=True, exist_ok=True)
    h, w, _ = original_rgb.shape

    # 01_original.png
    Image.fromarray(original_rgb).save(hash_dir / "01_original.png")

    # 02_depth.png
    d_min, d_max = depth_map.min(), depth_map.max()
    depth_vis = ((depth_map - d_min) / max(1e-5, d_max - d_min) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(depth_vis).save(hash_dir / "02_depth.png")

    # 03_candidate_contact_sheet.png
    cols = 3
    num_cand = len(features_list)
    rows = (num_cand + cols - 1) // cols if num_cand > 0 else 1
    panel_w = 320
    panel_h = int(panel_w * (h / float(w)))

    panels = []
    gray_bg = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    base_vis = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2RGB)

    score_map = {s.candidate_id: s for s in scores_list}
    sel_ids = selected_group.candidate_ids

    for feat in features_list:
        cid = feat.candidate_id
        is_sel = cid in sel_ids
        score = score_map[cid]

        panel = base_vis.copy()
        color = np.array([0, 255, 0], dtype=np.uint8) if is_sel else np.array([0, 255, 255], dtype=np.uint8)

        if mask_by_id and cid in mask_by_id:
            cand_m = mask_by_id[cid]
            panel[cand_m] = (panel[cand_m] * 0.4 + color * 0.6).astype(np.uint8)

        cv2.rectangle(panel, (feat.bbox[1], feat.bbox[0]), (feat.bbox[3], feat.bbox[2]), (0, 255, 0) if is_sel else (0, 255, 255), 2)
        panel_resized = cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

        cv2.rectangle(panel_resized, (0, 0), (panel_w, 55), (0, 0, 0), -1)
        status_str = "SELECTED" if is_sel else "REJECTED"
        cv2.putText(panel_resized, f"Candidate #{cid:02d}: {status_str}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0) if is_sel else (0, 200, 255), 1)
        cv2.putText(panel_resized, f"Score: {score.final_score:.2f} (Area: {feat.mask_area_ratio:.1%})", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(panel_resized, f"Depth: {feat.foreground_depth_mean:.2f} Contam: {feat.background_contamination_score:.2f}", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        panels.append(panel_resized)

    blank_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    while len(panels) < rows * cols:
        panels.append(blank_panel)

    grid_rows = [np.hstack(panels[r * cols : (r + 1) * cols]) for r in range(rows)]
    contact_sheet = np.vstack(grid_rows)
    Image.fromarray(contact_sheet).save(hash_dir / "03_candidate_contact_sheet.png")
    Image.fromarray(contact_sheet).save(hash_dir / "candidate_masks_contact_sheet.png")

    # 04_candidate_scores.png (Visual plot bar of top scores)
    plot_w, plot_h = 640, 320
    score_plot = np.full((plot_h, plot_w, 3), fill_value=245, dtype=np.uint8)
    cv2.putText(score_plot, "Candidate Multi-Signal Final Scores", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    for i, s in enumerate(scores_list[:min(10, len(scores_list))]):
        bar_len = int((s.final_score / 1.0) * (plot_w - 200))
        y_pos = 50 + i * 25
        cv2.rectangle(score_plot, (150, y_pos), (150 + bar_len, y_pos + 18), (0, 200, 0) if s.candidate_id in sel_ids else (200, 100, 0), -1)
        cv2.putText(score_plot, f"Cand #{s.candidate_id:02d}", (15, y_pos + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(score_plot, f"{s.final_score:.3f}", (155 + bar_len, y_pos + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    Image.fromarray(score_plot).save(hash_dir / "04_candidate_scores.png")

    # 05_candidate_groups.png & 06_selected_group.png
    raw_mask = selected_group.merged_mask
    Image.fromarray((raw_mask * 255).astype(np.uint8)).save(hash_dir / "05_candidate_groups.png")
    Image.fromarray((raw_mask * 255).astype(np.uint8)).save(hash_dir / "06_selected_group.png")

    # 07_raw_subject_mask.png & 08_refined_subject_mask.png
    Image.fromarray((raw_mask * 255).astype(np.uint8)).save(hash_dir / "07_raw_subject_mask.png")
    Image.fromarray((refined_mask * 255).astype(np.uint8)).save(hash_dir / "08_refined_subject_mask.png")

    # 09_subject_overlay.png
    overlay_rgb = original_rgb.copy()
    overlay_rgb[refined_mask] = (overlay_rgb[refined_mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5).astype(np.uint8)
    sub_contours, _ = cv2.findContours((refined_mask.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_rgb, sub_contours, -1, (0, 255, 255), 2)
    Image.fromarray(overlay_rgb).save(hash_dir / "09_subject_overlay.png")

    # 10_depth_subject_overlay.png
    depth_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
    depth_rgb[refined_mask] = (depth_rgb[refined_mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5).astype(np.uint8)
    Image.fromarray(depth_rgb).save(hash_dir / "10_depth_subject_overlay.png")

    # 11_background_contamination.png
    contam_vis = np.zeros_like(original_rgb)
    primary_feat = features_list[0] if features_list else None
    if primary_feat and primary_feat.background_contamination_score > 0.1:
        contam_vis[:, :] = [0, 0, int(255 * primary_feat.background_contamination_score)]
    Image.fromarray(contam_vis).save(hash_dir / "11_background_contamination.png")

    # 12_validation_report.json
    candidate_breakdown = []
    for feat in features_list:
        score = score_map[feat.candidate_id]
        candidate_breakdown.append({
            "candidate_id": feat.candidate_id,
            "sam_confidence": score.sam_confidence,
            "centrality_score": score.centrality_score,
            "depth_saliency_score": score.depth_saliency_score,
            "scale_score": score.scale_score,
            "relative_prominence_score": score.relative_prominence_score,
            "compound_support_score": score.compound_support_score,
            "border_penalty": score.border_penalty,
            "contamination_penalty": score.contamination_penalty,
            "environmental_penalty": score.environmental_penalty,
            "raw_score": score.raw_score,
            "final_score": score.final_score,
            "is_selected": feat.candidate_id in sel_ids
        })

    val_report = {
        "validation_status": validation_result.validation_status,
        "is_valid": validation_result.is_valid,
        "status_reasons": validation_result.status_reasons,
        "confidence": {
            "sam_model_confidence": validation_result.confidence.sam_model_confidence,
            "geometric_confidence": validation_result.confidence.geometric_confidence,
            "depth_confidence": validation_result.confidence.depth_confidence,
            "semantic_subject_confidence": validation_result.confidence.semantic_subject_confidence,
            "final_subject_confidence": validation_result.confidence.final_subject_confidence
        },
        "selected_group": {
            "group_id": selected_group.group_id,
            "candidate_ids": selected_group.candidate_ids,
            "final_score": selected_group.combined_score.final_score
        },
        "candidate_scores_breakdown": candidate_breakdown,
        "total_candidates_evaluated": len(features_list),
        "total_groups_evaluated": len(groups)
    }

    with open(hash_dir / "12_validation_report.json", "w") as f:
        json.dump(val_report, f, indent=2)
    with open(hash_dir / "candidate_selection.json", "w") as f:
        json.dump(val_report, f, indent=2)


def select_semantic_subject(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    predictor: SAM2ImagePredictor,
    hash_dir: Optional[Path] = None,
    config: SubjectSelectionConfig = SubjectSelectionConfig()
) -> SubjectSelectionResult:
    """
    Main entry point for semantic subject selection.
    Executes:
    1. SAM 2 Candidate Mask Generation
    2. Feature Extraction
    3. Multi-Signal Scoring & Ranking
    4. Candidate Grouping (handling compound subjects like Vishnu + Shesha)
    5. Mask Union & Edge-Preserving Refinement
    6. Mask Acceptance Gate Validation & Semantic Confidence Model
    7. Diagnostics Artifact Export

    Raises ValueError if validation gate REJECTS or UNCERTAIN before Phase C background reconstruction.
    """
    h, w = depth_map.shape

    # 1. Candidate Generation
    raw_masks = generate_candidate_masks_sam2(rgb_array, depth_map, predictor)

    if not raw_masks:
        raise ValueError("SUBJECT SELECTION FAILURE: SAM 2 generated zero valid candidate masks.")

    # 2. Candidate Feature Extraction
    features_list: List[CandidateFeatures] = []
    mask_by_id: Dict[int, np.ndarray] = {}

    for idx, cand_dict in enumerate(raw_masks, start=1):
        mask_bool = cand_dict["mask_bool"]
        sam_score = cand_dict["sam_score"]
        origin = cand_dict["prompt_origin"]

        feat = extract_candidate_features(
            mask_bool,
            sam_score=sam_score,
            prompt_origin=origin,
            depth_map=depth_map,
            rgb_array=rgb_array,
            candidate_id=idx,
            config=config
        )
        features_list.append(feat)
        mask_by_id[idx] = mask_bool

    # Compute relative batch features (prominence, foreground cluster distance, compound support, environmental isolation)
    features_list = compute_batch_relative_features(features_list, mask_by_id, depth_map, config)

    scores_list = [score_candidate_features(f, config) for f in features_list]
    features_list, scores_list = rank_candidates(features_list, scores_list)

    # 3. Candidate Grouping
    groups = generate_candidate_groups(
        mask_by_id,
        features_list,
        scores_list,
        depth_map,
        rgb_array,
        config=config
    )

    selected_group = groups[0] if groups else None

    if selected_group is None:
        raise ValueError("SUBJECT SELECTION FAILURE: Zero valid candidate groups evaluated.")

    # 4. Mask Refinement
    refined_mask = refine_subject_mask(selected_group.merged_mask, rgb_array)

    # 5. Mask Acceptance Gate & Semantic Subject Confidence
    score_margin = (
        groups[0].combined_score.final_score - groups[1].combined_score.final_score
        if len(groups) > 1 else groups[0].combined_score.final_score
    )
    features_map = {f.candidate_id: f for f in features_list}
    val_result = validate_selected_subject_mask(selected_group, features_map, score_margin, config)

    # 6. Export Diagnostics if hash_dir provided
    if hash_dir is not None:
        export_diagnostics(
            hash_dir, rgb_array, depth_map, features_list, scores_list, groups,
            selected_group, refined_mask, val_result, mask_by_id=mask_by_id
        )

    if not val_result.is_valid:
        raise ValueError(f"SUBJECT SELECTION: AMBIGUOUS - {'; '.join(val_result.status_reasons)}")

    return SubjectSelectionResult(
        selected_mask=selected_group.merged_mask,
        refined_mask=refined_mask,
        selected_group_ids=selected_group.candidate_ids,
        candidate_features_list=features_list,
        candidate_scores_list=scores_list,
        candidate_groups=groups,
        validation_result=val_result,
        metrics_summary={
            "selected_group_id": selected_group.group_id,
            "final_subject_confidence": val_result.confidence.final_subject_confidence,
            "validation_status": val_result.validation_status
        }
    )
