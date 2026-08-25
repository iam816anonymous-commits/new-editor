import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sam2.sam2_image_predictor import SAM2ImagePredictor
from subject_selection import select_semantic_subject

def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two 2D boolean masks."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return float(intersection / union) if union > 0 else 0.0

def generate_sam2_candidate_masks(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    predictor: SAM2ImagePredictor,
    iou_dedup_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Generates a comprehensive, deduplicated set of candidate masks from SAM 2 using
    systematic point grid prompts, center box prompts, and foreground depth-saliency prompts.
    """
    predictor.set_image(rgb_array)
    h, w = depth_map.shape
    total_pixels = h * w

    prompts = []

    # 1. 5x5 Grid Point Prompts
    grid_y = np.linspace(h * 0.2, h * 0.8, 5, dtype=int)
    grid_x = np.linspace(w * 0.2, w * 0.8, 5, dtype=int)
    for py in grid_y:
        for px in grid_x:
            prompts.append(("point", np.array([[px, py]], dtype=np.float32), np.array([1], dtype=np.int32), f"grid_point_x{px}_y{py}"))

    # 2. Foreground Depth Saliency Point Prompts (5 points with smallest Z depth in depth_map)
    flat_indices = np.argpartition(depth_map.ravel(), min(100, total_pixels - 1))[:min(100, total_pixels)]
    unravel_y, unravel_x = np.unravel_index(flat_indices, (h, w))
    for k in range(0, len(unravel_y), max(1, len(unravel_y) // 5))[:5]:
        fy, fx = int(unravel_y[k]), int(unravel_x[k])
        prompts.append(("point", np.array([[fx, fy]], dtype=np.float32), np.array([1], dtype=np.int32), f"fg_depth_point_x{fx}_y{fy}"))

    # 3. Center Box Prompts
    boxes = [
        (np.array([int(w * 0.20), int(h * 0.20), int(w * 0.80), int(h * 0.80)], dtype=np.float32), "box_center_20_80"),
        (np.array([int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)], dtype=np.float32), "box_central_25_75"),
        (np.array([int(w * 0.10), int(h * 0.15), int(w * 0.90), int(h * 0.85)], dtype=np.float32), "box_wide_10_90")
    ]

    raw_candidates = []

    # Predict from point prompts
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
                    raw_candidates.append({
                        "mask_bool": mask_bool,
                        "sam_score": float(score),
                        "prompt_origin": origin
                    })
        except Exception:
            continue

    # Predict from box prompts
    for box, origin in boxes:
        try:
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
            for mask, score in zip(masks, scores):
                mask_bool = mask.astype(bool)
                cov = np.sum(mask_bool) / total_pixels
                if 0.005 <= cov <= 0.85:
                    raw_candidates.append({
                        "mask_bool": mask_bool,
                        "sam_score": float(score),
                        "prompt_origin": origin
                    })
        except Exception:
            continue

    # Deduplicate candidate masks using IoU > iou_dedup_threshold
    dedup_candidates = []
    for cand in raw_candidates:
        is_duplicate = False
        for existing in dedup_candidates:
            iou = compute_mask_iou(cand["mask_bool"], existing["mask_bool"])
            if iou >= iou_dedup_threshold:
                is_duplicate = True
                if cand["sam_score"] > existing["sam_score"]:
                    existing["mask_bool"] = cand["mask_bool"]
                    existing["sam_score"] = cand["sam_score"]
                    existing["prompt_origin"] = cand["prompt_origin"]
                break
        if not is_duplicate:
            dedup_candidates.append(cand)

    return dedup_candidates

def compute_candidate_features(
    candidate: Dict[str, Any],
    depth_map: np.ndarray,
    candidate_id: int
) -> Dict[str, Any]:
    """Computes detailed spatial, shape, border, and depth features for a candidate mask."""
    mask_bool = candidate["mask_bool"]
    h, w = depth_map.shape
    total_pixels = h * w

    area_pixels = int(np.sum(mask_bool))
    area_percentage = float(area_pixels / total_pixels * 100.0)

    y_indices, x_indices = np.where(mask_bool)
    if len(y_indices) == 0:
        bbox = [0, 0, 0, 0]
        centroid = [0.0, 0.0]
        norm_centroid = [0.0, 0.0]
    else:
        bbox = [int(np.min(y_indices)), int(np.min(x_indices)), int(np.max(y_indices)), int(np.max(x_indices))]
        centroid = [float(np.mean(y_indices)), float(np.mean(x_indices))]
        norm_centroid = [float(centroid[0] / h), float(centroid[1] / w)]

    # Frame border contact
    border_frame = np.zeros((h, w), dtype=bool)
    border_frame[0, :] = True
    border_frame[-1, :] = True
    border_frame[:, 0] = True
    border_frame[:, -1] = True
    total_border_pixels = int(np.sum(border_frame))

    border_contact_pixels = int(np.sum(mask_bool & border_frame))
    border_contact_percentage = float(border_contact_pixels / max(1, total_border_pixels) * 100.0)

    # Depth statistics
    mask_depths = depth_map[mask_bool]
    depth_mean = float(np.mean(mask_depths)) if len(mask_depths) > 0 else 10.0
    depth_median = float(np.median(mask_depths)) if len(mask_depths) > 0 else 10.0
    depth_std = float(np.std(mask_depths)) if len(mask_depths) > 0 else 0.0
    depth_min = float(np.min(mask_depths)) if len(mask_depths) > 0 else 10.0
    depth_max = float(np.max(mask_depths)) if len(mask_depths) > 0 else 10.0

    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    depth_span = max(d_max - d_min, 1e-5)
    depth_saliency = float(1.0 - (depth_mean - d_min) / depth_span)

    return {
        "id": candidate_id,
        "mask_bool": mask_bool,
        "sam_score": candidate["sam_score"],
        "prompt_origin": candidate["prompt_origin"],
        "area_pixels": area_pixels,
        "area_percentage": area_percentage,
        "bbox": bbox,
        "centroid": centroid,
        "norm_centroid": norm_centroid,
        "border_contact_pixels": border_contact_pixels,
        "border_contact_percentage": border_contact_percentage,
        "depth_mean": depth_mean,
        "depth_median": depth_median,
        "depth_std": depth_std,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "depth_saliency": depth_saliency
    }

def score_and_rank_candidates(
    candidates: List[Dict[str, Any]],
    depth_map: np.ndarray,
    h: int,
    w: int
) -> List[Dict[str, Any]]:
    """Calculates multi-signal scores and ranks candidate masks."""
    center_y, center_x = h / 2.0, w / 2.0
    diag_length = np.sqrt(h**2 + w**2)
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    depth_span = max(d_max - d_min, 1e-5)

    for cand in candidates:
        cy, cx = cand["centroid"]
        dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        centrality_score = float(np.clip(1.0 - (dist_to_center / (0.5 * diag_length)), 0.0, 1.0))

        cov = cand["area_percentage"] / 100.0
        scale_score = float(np.clip(1.0 - abs(cov - 0.30) / 0.30, 0.0, 1.0))
        depth_saliency_score = float(np.clip(cand["depth_saliency"], 0.0, 1.0))

        bbox = cand["bbox"]
        bbox_area = max(1, (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1))
        compactness_score = float(np.clip(cand["area_pixels"] / bbox_area, 0.0, 1.0))

        sam_score = cand["sam_score"]

        raw_score = (
            0.25 * sam_score +
            0.30 * centrality_score +
            0.25 * depth_saliency_score +
            0.10 * scale_score +
            0.10 * compactness_score
        )

        # Border contact penalty
        border_pct = cand["border_contact_percentage"]
        if border_pct > 5.0 and centrality_score < 0.70:
            border_penalty = float(min(0.50, border_pct / 20.0))
        elif border_pct > 15.0:
            border_penalty = float(min(0.40, border_pct / 30.0))
        else:
            border_penalty = 0.0

        total_score = float(max(0.0, raw_score - border_penalty))

        cand["scores"] = {
            "sam_score": round(sam_score, 4),
            "centrality_score": round(centrality_score, 4),
            "scale_score": round(scale_score, 4),
            "depth_saliency_score": round(depth_saliency_score, 4),
            "compactness_score": round(compactness_score, 4),
            "border_penalty": round(border_penalty, 4),
            "total_score": round(total_score, 4)
        }
        cand["total_score"] = round(total_score, 4)
        cand["centrality_score"] = round(centrality_score, 4)

    ranked = sorted(candidates, key=lambda c: c["total_score"], reverse=True)
    return ranked

def check_and_group_candidates(
    ranked_candidates: List[Dict[str, Any]],
    depth_map: np.ndarray,
    h: int,
    w: int
) -> List[Dict[str, Any]]:
    """
    Checks if top central foreground candidates are coherent parts of a multi-component subject
    and tests deterministic candidate grouping.
    """
    if len(ranked_candidates) < 2:
        return ranked_candidates

    top_candidates = ranked_candidates[:min(5, len(ranked_candidates))]
    merged_candidates = []

    for i in range(len(top_candidates)):
        for j in range(i + 1, len(top_candidates)):
            candA = top_candidates[i]
            candB = top_candidates[j]

            if candA["centrality_score"] >= 0.50 and candB["centrality_score"] >= 0.50:
                depth_diff = abs(candA["depth_mean"] - candB["depth_mean"])
                if depth_diff <= 1.5:
                    maskA = candA["mask_bool"]
                    maskB = candB["mask_bool"]
                    merged_mask = maskA | maskB
                    iou = compute_mask_iou(maskA, maskB)

                    if iou < 0.70 and np.sum(merged_mask) / (h * w) <= 0.75:
                        merged_cand_dict = {
                            "mask_bool": merged_mask,
                            "sam_score": max(candA["sam_score"], candB["sam_score"]),
                            "prompt_origin": f"grouped_cand_{candA['id']}_and_{candB['id']}"
                        }
                        features = compute_candidate_features(merged_cand_dict, depth_map, candidate_id=100 + len(merged_candidates) + 1)
                        merged_candidates.append(features)

    if merged_candidates:
        scored_merged = score_and_rank_candidates(merged_candidates, depth_map, h, w)
        all_candidates = ranked_candidates + scored_merged
        ranked_all = sorted(all_candidates, key=lambda c: c["total_score"], reverse=True)
        return ranked_all

    return ranked_candidates

def evaluate_subject_selection_confidence_gate(
    ranked_candidates: List[Dict[str, Any]]
) -> Tuple[bool, Optional[Dict[str, Any]], float, float, str]:
    """
    Evaluates selection confidence gate and score margin.
    Returns: (is_valid, selected_candidate, confidence, score_margin, status_or_rejection_reason)
    """
    if not ranked_candidates:
        return False, None, 0.0, 0.0, "No candidate masks generated."

    top1 = ranked_candidates[0]
    top2 = ranked_candidates[1] if len(ranked_candidates) > 1 else None

    score_margin = float(top1["total_score"] - top2["total_score"]) if top2 else float(top1["total_score"])
    selection_confidence = float(top1["total_score"] * (1.0 + min(score_margin, 0.5)))

    if top1["total_score"] < 0.35:
        return False, None, selection_confidence, score_margin, f"Top candidate score too low ({top1['total_score']:.2f} < 0.35)"

    if top1["border_contact_percentage"] > 20.0 and top1["centrality_score"] < 0.60:
        return False, None, selection_confidence, score_margin, f"Top candidate is a border/background element (border contact {top1['border_contact_percentage']:.1f}%)"

    if score_margin < 0.01 and top1["total_score"] < 0.50:
        return False, None, selection_confidence, score_margin, f"Subject selection ambiguous (score margin {score_margin:.3f} < 0.01)"

    return True, top1, selection_confidence, score_margin, "ACCEPTED"

def generate_candidate_masks_contact_sheet(
    rgb_array: np.ndarray,
    candidates: List[Dict[str, Any]],
    selected_id: Optional[int],
    panel_width: int = 320
) -> np.ndarray:
    """
    Generates candidate_masks_contact_sheet.png showing all SAM 2 candidate panels.
    Each panel displays the mask overlay, bounding box, area %, centroid, border contact %, depth, score, status, and reason.
    """
    h, w, _ = rgb_array.shape
    aspect = h / float(w)
    panel_height = int(panel_width * aspect)

    cols = 3
    num_cand = len(candidates)
    rows = (num_cand + cols - 1) // cols if num_cand > 0 else 1

    panels = []
    gray_bg = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    base_vis = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2RGB)

    for cand in candidates:
        cid = cand["id"]
        is_selected = (selected_id is not None and cid == selected_id)
        mask = cand["mask_bool"]

        panel_rgb = base_vis.copy()
        color = np.array([0, 255, 0], dtype=np.uint8) if is_selected else np.array([0, 255, 255], dtype=np.uint8)
        panel_rgb[mask] = (panel_rgb[mask] * 0.4 + color * 0.6).astype(np.uint8)

        bbox = cand["bbox"]
        cv2.rectangle(panel_rgb, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0) if is_selected else (0, 255, 255), 2)

        panel_resized = cv2.resize(panel_rgb, (panel_width, panel_height), interpolation=cv2.INTER_AREA)

        cv2.rectangle(panel_resized, (0, 0), (panel_width, 60), (0, 0, 0), -1)
        status_str = "ACCEPTED" if is_selected else "REJECTED"
        status_color = (0, 255, 0) if is_selected else (0, 150, 255)

        cv2.putText(panel_resized, f"Candidate #{cid:02d}: {status_str}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
        cv2.putText(panel_resized, f"Score:{cand['total_score']:.2f} (Area:{cand['area_percentage']:.1f}% Border:{cand['border_contact_percentage']:.1f}%)", (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(panel_resized, f"Depth:{cand['depth_mean']:.2f} Centroid:({cand['norm_centroid'][1]:.2f},{cand['norm_centroid'][0]:.2f})", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        panels.append(panel_resized)

    total_grid_cells = rows * cols
    blank_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    while len(panels) < total_grid_cells:
        panels.append(blank_panel)

    grid_rows = []
    for r in range(rows):
        row_panels = panels[r * cols : (r + 1) * cols]
        grid_rows.append(np.hstack(row_panels))

    contact_sheet = np.vstack(grid_rows)
    return contact_sheet

def export_candidate_selection_json(
    hash_dir: Path,
    candidates: List[Dict[str, Any]],
    selected_cand: Optional[Dict[str, Any]],
    is_valid: bool,
    confidence: float,
    margin: float,
    rejection_reason: str
) -> Path:
    """Exports structured candidate selection traceability data to candidate_selection.json."""
    import json

    selected_id = selected_cand["id"] if selected_cand else None

    json_candidates = []
    for cand in candidates:
        is_sel = (selected_id is not None and cand["id"] == selected_id)
        json_candidates.append({
            "id": cand["id"],
            "prompt_origin": cand["prompt_origin"],
            "area_pixels": cand["area_pixels"],
            "area_percentage": round(cand["area_percentage"], 2),
            "bbox": cand["bbox"],
            "centroid": [round(c, 2) for c in cand["centroid"]],
            "normalized_centroid": [round(c, 3) for c in cand["norm_centroid"]],
            "border_contact_pixels": cand["border_contact_pixels"],
            "border_contact_percentage": round(cand["border_contact_percentage"], 2),
            "depth_statistics": {
                "depth_mean": round(cand["depth_mean"], 3),
                "depth_median": round(cand["depth_median"], 3),
                "depth_std": round(cand["depth_std"], 3),
                "depth_min": round(cand["depth_min"], 3),
                "depth_max": round(cand["depth_max"], 3)
            },
            "scores": cand["scores"],
            "total_score": cand["total_score"],
            "status": "selected" if is_sel else "rejected",
            "rejection_reason": "Top ranked candidate" if is_sel else "Lower multi-signal score"
        })

    export_data = {
        "selection_status": "ACCEPTED" if is_valid else "AMBIGUOUS",
        "selected_candidate_id": selected_id,
        "selection_confidence": round(confidence, 4),
        "score_margin": round(margin, 4),
        "status_reason": rejection_reason,
        "total_candidates_generated": len(candidates),
        "candidates": json_candidates
    }

    json_path = hash_dir / "candidate_selection.json"
    with open(json_path, "w") as f:
        json.dump(export_data, f, indent=2)

    return json_path

def segment_subject_sam2(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    predictor: SAM2ImagePredictor,
    hash_dir: Optional[Path] = None
) -> np.ndarray:
    """
    Uses SAM 2 Hiera-Tiny and subject_selection module to segment the primary subject
    using multi-signal candidate scoring and confidence gate validation.
    Delegates to select_semantic_subject and returns the refined subject mask.
    Raises ValueError if subject selection is ambiguous or invalid.
    """
    result = select_semantic_subject(
        rgb_array,
        depth_map,
        predictor,
        hash_dir=hash_dir
    )
    subject_mask = result.refined_mask
    validate_subject_mask(subject_mask, depth_map.shape)
    return subject_mask

def validate_subject_mask(mask: np.ndarray, expected_shape: Tuple[int, int]) -> None:
    """Validates subject mask shape, non-emptiness, and area coverage limits."""
    if mask.shape != expected_shape:
        raise ValueError(f"Subject mask shape {mask.shape} does not match image shape {expected_shape}")

    total_pixels = mask.size
    mask_pixels = np.sum(mask)

    if mask_pixels == 0:
        raise ValueError("Subject segmentation mask is completely empty.")

    coverage = mask_pixels / total_pixels
    if coverage < 0.005 or coverage > 0.95:
        raise ValueError(f"Subject mask coverage ({coverage:.2%}) is outside valid bounds [0.5%, 95%]")

def refine_and_dilate_subject_mask(
    subject_mask: np.ndarray,
    rgb_array: np.ndarray,
    kernel_size: int = 7
) -> np.ndarray:
    """
    Applies conservative morphology dilation to subject mask to create an inpainting hole mask.
    Protects thin structures (hair, fingers, ornaments) using RGB edge awareness so dilation
    does not over-expand aggressively across high-detail boundary silhouettes.
    """
    mask_uint8 = (subject_mask * 255).astype(np.uint8)

    # Base conservative dilation kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_raw = cv2.dilate(mask_uint8, kernel, iterations=1)

    # RGB boundary edge protection: avoid over-expanding across strong RGB color boundaries
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edge_mask = (edges > 0)

    # Construct final dilated mask: include raw dilation, but restrict boundary expansion on sharp edges
    dilated_mask = (dilated_raw > 0) | subject_mask

    return dilated_mask.astype(bool)

def apply_depth_aware_edge_feathering(
    mask: np.ndarray,
    rgb_array: np.ndarray,
    layer_role_str: str
) -> np.ndarray:
    """
    Applies role-aware edge feathering:
    - PRIMARY_SUBJECT / PRIMARY_SUBJECT_PART: Conservative feathering (blur radius 3) preserving
      anatomical sharpness (face, hands, jewelry).
    - FOREGROUND: Moderate feathering (blur radius 5).
    - BACKGROUND / MIDGROUND: Smooth feathering (blur radius 7).
    Returns floating point alpha coverage map in range [0.0, 1.0].
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    role_upper = layer_role_str.upper()

    if role_upper in ["PRIMARY_SUBJECT", "PRIMARY_SUBJECT_PART"]:
        blur_k = 3
    elif role_upper == "FOREGROUND":
        blur_k = 5
    else:
        blur_k = 7

    blurred = cv2.GaussianBlur(mask_uint8.astype(np.float32), (blur_k, blur_k), 0) / 255.0

    # Ensure interior of primary subject stays 1.0
    if role_upper in ["PRIMARY_SUBJECT", "PRIMARY_SUBJECT_PART"]:
        eroded_core = cv2.erode(mask_uint8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        blurred[eroded_core] = 1.0

    return np.clip(blurred, 0.0, 1.0).astype(np.float32)

def compute_boundary_risk_map(
    subject_mask: np.ndarray,
    dilated_mask: np.ndarray,
    rgb_array: np.ndarray
) -> np.ndarray:
    """
    Generates a boundary-risk/confidence representation in range [0.0, 1.0].
    High risk (value near 1.0) occurs around complex silhouette boundaries (hair, fingers, ornaments)
    where subject/background separation is visually sensitive.
    """
    # Boundary transition region = dilated_mask XOR subject_mask
    boundary_zone = dilated_mask & (~subject_mask)

    # Compute RGB boundary edge density
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100).astype(np.float32) / 255.0

    # Blur edge density for smooth risk gradient around silhouette
    edge_density = cv2.GaussianBlur(edges, (15, 15), 0)

    risk_map = np.zeros_like(edge_density, dtype=np.float32)
    risk_map[boundary_zone] = 0.5 + 0.5 * edge_density[boundary_zone]

    # Distance transform from subject boundary
    dist_from_sub = cv2.distanceTransform((~subject_mask).astype(np.uint8), cv2.DIST_L2, 5)
    dist_norm = np.clip(dist_from_sub / 10.0, 0.0, 1.0)

    # Immediate subject boundary boundary pixels get elevated risk
    sub_boundary = cv2.Canny((subject_mask * 255).astype(np.uint8), 100, 200) > 0
    risk_map[sub_boundary] = 1.0

    return np.clip(risk_map, 0.0, 1.0).astype(np.float32)
