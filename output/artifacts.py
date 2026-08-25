import cv2
import json
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Dict, Any, Optional
from quality.diagnostics import generate_subject_coherence_diagnostics, generate_phase_e_keyframe_contact_sheet

def compute_image_sha256(image_path: Path) -> str:
    """Computes SHA-256 hash of the input image file."""
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    sha256 = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def validate_and_load_image(image_path: Path) -> Tuple[Image.Image, np.ndarray, str]:
    """Validates and loads an input image. Returns (PIL Image, RGB numpy array, short_hash)."""
    try:
        pil_img = Image.open(image_path).convert("RGB")
        rgb_array = np.array(pil_img)
    except Exception as e:
        raise ValueError(f"Failed to decode image at '{image_path}': {e}") from e

    full_hash = compute_image_sha256(image_path)
    short_hash = full_hash[:8]
    return pil_img, rgb_array, short_hash

def setup_cache_directory(base_cache_dir: Path, short_hash: str) -> Path:
    """Creates SHA-256 content-addressed cache directory cache/<short_hash>/."""
    cache_dir = base_cache_dir / short_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def setup_output_directories(base_dir: Path, short_hash: str, create_subdirs: bool = True) -> Path:
    """Creates directory structure output/<short_hash>/."""
    hash_dir = base_dir / short_hash
    hash_dir.mkdir(parents=True, exist_ok=True)

    if create_subdirs:
        for level in ["subtle", "cinematic", "strong"]:
            (hash_dir / level).mkdir(parents=True, exist_ok=True)

    return hash_dir

def save_phase_b_diagnostic_artifacts(
    hash_dir: Path,
    depth_map: np.ndarray,
    subject_mask: np.ndarray,
    confidence_map: np.ndarray
) -> None:
    """Saves depth.png, subject_mask.png, and confidence_map.png to output/<short_hash>/."""
    # 1. Depth visualization (normalized 0..255 grayscale / inferno visualization)
    d_min, d_max = depth_map.min(), depth_map.max()
    depth_vis = ((depth_map - d_min) / (d_max - d_min) * 255.0).astype(np.uint8) if d_max > d_min else np.zeros_like(depth_map, dtype=np.uint8)
    Image.fromarray(depth_vis).save(hash_dir / "depth.png")

    # 2. Subject mask visualization (0 or 255)
    mask_vis = (subject_mask * 255).astype(np.uint8)
    Image.fromarray(mask_vis).save(hash_dir / "subject_mask.png")

    # 3. Confidence map visualization (0..255)
    conf_vis = (confidence_map * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(conf_vis).save(hash_dir / "confidence_map.png")

def save_phase_c_diagnostic_artifacts(
    hash_dir: Path,
    background_plate: np.ndarray,
    background_depth: np.ndarray,
    provenance_map: np.ndarray,
    boundary_risk_map: np.ndarray
) -> None:
    """Saves Phase C diagnostic artifacts to output/<short_hash>/."""
    # 1. Clean RGB Background Plate
    Image.fromarray(background_plate).save(hash_dir / "background_plate.png")

    # 2. Background Depth Visualization
    d_min, d_max = background_depth.min(), background_depth.max()
    depth_vis = ((background_depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8) if d_max > d_min else np.zeros_like(background_depth, dtype=np.uint8)
    Image.fromarray(depth_vis).save(hash_dir / "background_depth.png")

    # 3. Provenance Map (255 for OBSERVED, 0 for RECONSTRUCTED)
    prov_vis = (provenance_map * 255.0).astype(np.uint8)
    Image.fromarray(prov_vis).save(hash_dir / "provenance_map.png")

    # 4. Boundary Risk Map (0..255)
    risk_vis = (boundary_risk_map * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(risk_vis).save(hash_dir / "boundary_risk_map.png")

def save_phase_d_diagnostic_artifacts(
    hash_dir: Path,
    zero_motion_rgb: np.ndarray,
    zero_motion_diff: np.ndarray,
    micro_motion_rgb: np.ndarray,
    micro_motion_diff: np.ndarray,
    micro_motion_prov: np.ndarray,
    crop_diagnostics: np.ndarray
) -> None:
    """Saves Phase D diagnostic artifacts to output/<short_hash>/."""
    Image.fromarray(zero_motion_rgb).save(hash_dir / "phase_d_zero_motion.png")
    Image.fromarray(zero_motion_diff).save(hash_dir / "phase_d_zero_motion_diff.png")
    Image.fromarray(micro_motion_rgb).save(hash_dir / "phase_d_micro_motion.png")
    Image.fromarray(micro_motion_diff).save(hash_dir / "phase_d_difference.png")
    Image.fromarray((micro_motion_prov * 255.0).astype(np.uint8)).save(hash_dir / "phase_d_provenance.png")
    Image.fromarray(crop_diagnostics).save(hash_dir / "phase_d_crop_diagnostics.png")

def save_phase_e_artifacts(
    hash_dir: Path,
    plan_summary: Dict[str, Any],
    keyframes: Dict[str, np.ndarray],
    keyframe_metrics: Dict[str, Dict[str, float]],
    translations: np.ndarray,
    rotations: np.ndarray,
    safety_margins: Dict[str, float],
    scaling_sweep: Dict[float, Dict[str, float]],
    subject_mask: np.ndarray,
    original_rgb: np.ndarray
) -> None:
    """Saves motion_plan.json, representative keyframes, contact sheets, and trajectory diagnostic plots."""
    import json

    # Save motion_plan.json
    plan_file = hash_dir / "motion_plan.json"
    full_export = {
        "plan_summary": plan_summary,
        "safety_margins": safety_margins,
        "magnitude_scaling_sweep": {str(k): v for k, v in scaling_sweep.items()},
        "keyframe_metrics": keyframe_metrics
    }
    with open(plan_file, "w") as f:
        json.dump(full_export, f, indent=2)

    # Save 5 representative keyframe PNGs
    for name, img in keyframes.items():
        Image.fromarray(img).save(hash_dir / f"frame_{name}.png")

    # Save Subject Coherence Diagnostics Sheet
    coh_sheet = generate_subject_coherence_diagnostics(original_rgb, keyframes, subject_mask)
    Image.fromarray(coh_sheet).save(hash_dir / "phase_e_subject_coherence_diagnostics.png")

    # Save Keyframe Visual Contact Sheet
    kf_sheet = generate_phase_e_keyframe_contact_sheet(original_rgb, keyframes, subject_mask)
    Image.fromarray(kf_sheet).save(hash_dir / "phase_e_keyframe_contact_sheet.png")

    # Generate and save motion_trajectory.png & safety_envelope.png plots using OpenCV
    plot_w, plot_h = 640, 320

    # 1. Motion Trajectory Plot (X, Y, Z translations over frames)
    traj_img = np.full((plot_h, plot_w, 3), fill_value=245, dtype=np.uint8)
    cv2.putText(traj_img, "Camera Trajectory (tx, ty, tz)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    num_pts = len(translations)
    x_coords = np.linspace(50, plot_w - 20, num_pts, dtype=int)

    for i in range(num_pts - 1):
        # Scale tx to plot height
        pt1_x = (x_coords[i], int(plot_h / 2 - translations[i, 0] * 1000))
        pt2_x = (x_coords[i+1], int(plot_h / 2 - translations[i+1, 0] * 1000))
        cv2.line(traj_img, pt1_x, pt2_x, (255, 0, 0), 2)  # Blue = tx

        pt1_z = (x_coords[i], int(plot_h / 2 - translations[i, 2] * 500))
        pt2_z = (x_coords[i+1], int(plot_h / 2 - translations[i+1, 2] * 500))
        cv2.line(traj_img, pt1_z, pt2_z, (0, 150, 0), 2)  # Green = tz

    Image.fromarray(traj_img).save(hash_dir / "motion_trajectory.png")

    # 2. Safety Envelope Plot
    env_img = np.full((plot_h, plot_w, 3), fill_value=245, dtype=np.uint8)
    cv2.putText(env_img, "Closed-Loop Safety Envelope & Ceiling", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    ceiling_y = int(plot_h - (plan_summary['disparity_ceiling_target_px'] / (plot_summary_scale := 100.0)) * plot_h)
    cv2.line(env_img, (50, 150), (plot_w - 20, 150), (0, 0, 255), 2)  # Red ceiling line
    cv2.putText(env_img, f"Disparity Ceiling: {plan_summary['disparity_ceiling_target_px']:.1f}px", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    Image.fromarray(env_img).save(hash_dir / "safety_envelope.png")
    Image.fromarray(env_img).save(hash_dir / "trajectory_diagnostics.png")

def save_phase_d_validation_artifacts(
    hash_dir: Path,
    zero_motion_rgb: np.ndarray,
    zero_motion_diff: np.ndarray,
    zero_motion_error_mask: np.ndarray,
    discontinuity_rejection_map: np.ndarray,
    micro_sweep_contact_sheet: np.ndarray,
    sweep_frames: Dict[float, np.ndarray]
) -> None:
    """Saves Phase D Validation diagnostic artifacts to output/<short_hash>/."""
    Image.fromarray(zero_motion_rgb).save(hash_dir / "phase_d_zero_motion.png")
    Image.fromarray(zero_motion_diff).save(hash_dir / "phase_d_zero_motion_diff.png")
    Image.fromarray(zero_motion_error_mask).save(hash_dir / "zero_motion_error_mask.png")
    Image.fromarray(discontinuity_rejection_map).save(hash_dir / "discontinuity_rejection_map.png")
    Image.fromarray(micro_sweep_contact_sheet).save(hash_dir / "phase_d_micro_sweep_comparison.png")

    for frac, frame in sweep_frames.items():
        filename = f"phase_d_micro_motion_{frac}.png"
        Image.fromarray(frame).save(hash_dir / filename)
