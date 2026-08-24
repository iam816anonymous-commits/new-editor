"""
Authoritative Visual Quality Benchmark Matrix Executor for First-Principles Cinematic 2.5D Renderer.

Executes all 36 benchmark matrix configurations (3 resolutions x 4 motions x 3 strengths)
deterministically and exports complete diagnostic reports and quality summary artifacts under
output/visual_quality_benchmark/.
"""

import os
import json
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import v0_pipeline as v0

def run_benchmark_matrix():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=str, default="ALL", choices=["1024x683", "1536x1024", "1920x1080", "ALL"])
    args = parser.parse_args()

    print("=== First-Principles Cinematic 2.5D Renderer — Visual Quality Benchmark Matrix ===")

    os.makedirs("output/visual_quality_benchmark", exist_ok=True)
    img_orig = Image.open("test_assets/test.jpeg").convert("RGB")

    all_res = [
        (1024, 683),
        (1536, 1024),
        (1920, 1080)
    ]

    if args.res == "1024x683":
        resolutions = [(1024, 683)]
    elif args.res == "1536x1024":
        resolutions = [(1536, 1024)]
    elif args.res == "1920x1080":
        resolutions = [(1920, 1080)]
    else:
        resolutions = all_res

    motions = ["Cinematic Push-In", "Dolly Out", "Horizontal Pan", "Orbit"]
    strengths = ["LOW", "MEDIUM", "HIGH"]

    # Pre-create resized images
    res_images = {}
    for w, h in resolutions:
        resized = img_orig.resize((w, h), Image.Resampling.LANCZOS)
        img_path = Path(f"output/visual_quality_benchmark/input_{w}x{h}.jpg")
        resized.save(img_path)
        res_images[(w, h)] = img_path

    # Load models ONCE
    print("[*] Loading Depth Anything V2 Small and SAM 2 Hiera-Tiny models ONCE for benchmark session...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model, depth_processor = v0.load_depth_anything_v2(device)
    sam2_predictor = v0.load_sam2(device)
    print("[✓] Models loaded successfully!")

    matrix_results = []

    for w, h in resolutions:
        img_path = res_images[(w, h)]
        pil_img, rgb_array, short_hash = v0.validate_and_load_image(img_path)

        print(f"\n============================================================")
        print(f"[*] Processing Resolution Scene: {w}x{h} (Hash: {short_hash})")
        print(f"============================================================")

        # Depth Estimation
        raw_depth = v0.infer_raw_depth(pil_img, depth_model, depth_processor, device)
        norm_depth = v0.handle_depth_outliers_and_normalize(raw_depth)
        refined_depth = v0.edge_aware_depth_refinement(rgb_array, norm_depth)
        confidence_map = v0.compute_depth_confidence_map(refined_depth, rgb_array)

        # Subject Segmentation
        subject_mask = v0.segment_subject_sam2(rgb_array, refined_depth, sam2_predictor)

        # Background Reconstruction
        dilated_mask = cv2.dilate((subject_mask * 255).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
        b_risk = v0.compute_boundary_risk_map(subject_mask, dilated_mask, rgb_array)
        bg_plate = v0.reconstruct_background_rgb(rgb_array, dilated_mask)
        bg_depth = v0.complete_background_depth(refined_depth, dilated_mask)
        prov_map = v0.compute_provenance_map(dilated_mask)

        fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)

        for motion in motions:
            for strength in strengths:
                run_dir = Path(f"output/visual_quality_benchmark/{w}x{h}/{motion.lower().replace(' ', '_')}_{strength.lower()}/{short_hash}")
                run_dir.mkdir(parents=True, exist_ok=True)

                summary_json_path = run_dir / "quality_summary.json"
                if summary_json_path.exists() and summary_json_path.stat().st_size > 10:
                    print(f"[✓] Skipping (Already Executed): Resolution={w}x{h} | Motion={motion} | Amplitude={strength}")
                    with open(summary_json_path) as f:
                        matrix_results.append(json.load(f))
                    continue

                print(f"[*] Benchmark Run: Resolution={w}x{h} | Motion={motion} | Amplitude={strength}")

                # Plan trajectory
                trans_plan, rot_plan, final_scale, plan_summary = v0.plan_safe_motion_trajectory(
                    motion, strength, w, h, refined_depth, confidence_map, subject_mask, b_risk, prov_map,
                    fx, fy, cx, cy, num_frames=5
                )

                frames_dir = run_dir / "frames"
                disp_ceil = plan_summary["disparity_ceiling_target_px"]
                rendered_frames, per_frame_metrics = v0.render_full_frame_sequence(
                    rgb_array, refined_depth, bg_plate, bg_depth, prov_map, subject_mask, b_risk,
                    trans_plan, rot_plan, fx, fy, cx, cy, disp_ceil, frames_dir,
                    motion_amplitude=strength, frame_count=5
                )

                # Perceptual Motion & Visual Quality
                perceptual_diag = v0.compute_perceptual_motion_score(
                    rendered_frames, subject_mask, bg_depth, per_frame_metrics, trans_plan, rot_plan,
                    motion_amplitude=strength
                )

                from spatial_intelligence.visual_quality import compute_composite_quality_score
                visual_qual = compute_composite_quality_score(
                    rendered_frames, subject_mask, prov_map, bg_depth,
                    motion_effectiveness=perceptual_diag.get("motion_effectiveness_score", 0.80),
                    parallax_hierarchy=1.0 if perceptual_diag.get("motion_ordering_valid", True) else 0.5
                )

                # Export report
                run_report = {
                    "resolution": [w, h],
                    "requested_motion": motion,
                    "requested_amplitude": strength,
                    "achieved_amplitude": perceptual_diag.get("achieved_amplitude", strength),
                    "motion_visibility_class": perceptual_diag.get("motion_visibility_class", "VISIBLE"),
                    "motion_good": perceptual_diag.get("motion_good", True) and (visual_qual.overall_score >= 0.50),
                    "subject_displacement_px": perceptual_diag["image_space"]["subject_displacement_px"],
                    "foreground_displacement_px": perceptual_diag["image_space"]["foreground_displacement_px"],
                    "midground_displacement_px": perceptual_diag["image_space"]["midground_displacement_px"],
                    "background_displacement_px": perceptual_diag["image_space"]["background_displacement_px"],
                    "subject_scale_growth": perceptual_diag["image_space"]["subject_scale_growth"],
                    "overall_quality_score": visual_qual.overall_score,
                    "quality_class": visual_qual.quality_class,
                    "detected_artifact_codes": visual_qual.detected_artifact_codes,
                    "failure_reasons": perceptual_diag.get("failure_reasons", []) + visual_qual.failure_reasons
                }

                with open(run_dir / "quality_summary.json", "w") as f:
                    json.dump(run_report, f, indent=2)

                matrix_results.append(run_report)

    # Save complete matrix summary
    with open("output/visual_quality_benchmark/benchmark_matrix_summary.json", "w") as f:
        json.dump(matrix_results, f, indent=2)

    print(f"\n[✓] Visual Quality Benchmark Matrix Execution Complete! ({len(matrix_results)} runs executed)")

if __name__ == "__main__":
    run_benchmark_matrix()
