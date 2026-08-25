"""
High-Level Pipeline Application Execution Engine.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import time
import hashlib
import numpy as np
from PIL import Image

from rendering.video_encoder import verify_ffmpeg, encode_and_verify_mp4
from output.artifacts import (
    compute_image_sha256,
    validate_and_load_image,
    setup_output_directories,
    save_phase_b_diagnostic_artifacts,
    save_phase_c_diagnostic_artifacts,
)
from inference.device import get_device
from inference.model_manager import (
    load_depth_anything_v2,
    load_sam2,
    DEPTH_MODEL_ID,
    SAM2_MODEL_ID,
    SAM2_CKPT_FILENAME,
)
from inference.depth import (
    infer_raw_depth,
    handle_depth_outliers_and_normalize,
    edge_aware_depth_refinement,
    compute_depth_confidence_map,
)
from inference.segmentation import (
    validate_subject_mask,
    refine_and_dilate_subject_mask,
    compute_boundary_risk_map,
)
from subject_selection import select_semantic_subject
from rendering.disocclusion import (
    reconstruct_background_rgb,
    complete_background_depth,
    compute_provenance_map,
)
from spatial_intelligence.spatial_engine import analyze_spatial_scene
from camera.intrinsics import derive_camera_intrinsics
from camera.safety import plan_safe_motion_trajectory
from modes.router import select_render_mode
from modes.mode_2_5d.pipeline import Mode25DPipeline
from modes.mode_3d.pipeline import Mode3DPipeline
from core.contracts import RenderRequest

def run_pipeline(args) -> Dict[str, Any]:
    """Executes full cinematic view synthesis and diagnostic pipeline."""
    t_start_total = time.time()
    print("=== First-Principles Cinematic 2.5D/3D Parallax Renderer ===")

    # 1. Verify FFmpeg
    ffmpeg_version = verify_ffmpeg()
    print(f"[✓] FFmpeg detected: {ffmpeg_version}")

    # 2. Validate & Load Image
    input_path = Path(args.input)
    print(f"[*] Validating input image: {input_path}")
    pil_img, rgb_array, short_hash = validate_and_load_image(input_path)
    full_sha256 = compute_image_sha256(input_path)
    print(f"[✓] Image loaded successfully ({pil_img.width}x{pil_img.height}). SHA-256 Hash: {short_hash}")

    # 3. Setup Output Directory
    base_output = Path(args.output_dir)
    hash_dir = setup_output_directories(base_output, short_hash, create_subdirs=args.render_video)
    print(f"[✓] Output directory configured at: {hash_dir}")

    # Save original reference image copy
    original_save_path = hash_dir / "original.png"
    pil_img.save(original_save_path)

    # 4. Device & Model Loading
    device = get_device()
    print(f"[*] Compute Device selected: {device.upper()}")

    print("[*] Loading Depth Anything V2 Small model...")
    t_depth_load_start = time.time()
    depth_processor, depth_model = load_depth_anything_v2(device)
    t_depth_load = time.time() - t_depth_load_start

    print("[*] Loading SAM 2 Hiera-Tiny model...")
    t_sam2_load_start = time.time()
    sam2_predictor = load_sam2(device)
    t_sam2_load = time.time() - t_sam2_load_start

    # 5. Real Depth Inference & Processing
    print("[*] Performing REAL Depth Anything V2 monocular depth estimation...")
    t_depth_infer_start = time.time()
    raw_depth = infer_raw_depth(pil_img, depth_processor, depth_model, device)
    t_depth_infer = time.time() - t_depth_infer_start

    continuous_depth = handle_depth_outliers_and_normalize(raw_depth)
    refined_depth = edge_aware_depth_refinement(rgb_array, continuous_depth)
    confidence_map = compute_depth_confidence_map(refined_depth, rgb_array)

    # 6. SAM 2 Subject Segmentation
    print("[*] Performing REAL SAM 2 subject segmentation...")
    t_sam2_infer_start = time.time()
    sel_res = select_semantic_subject(rgb_array, refined_depth, sam2_predictor, hash_dir=hash_dir)
    subject_mask = sel_res.refined_mask
    validate_subject_mask(subject_mask, refined_depth.shape)
    t_sam2_infer = time.time() - t_sam2_infer_start

    # 7. Save Phase B Diagnostic Artifacts
    save_phase_b_diagnostic_artifacts(hash_dir, refined_depth, subject_mask, confidence_map)

    # 8. Disocclusion & Inpainting
    dilated_mask = refine_and_dilate_subject_mask(subject_mask, rgb_array)
    boundary_risk_map = compute_boundary_risk_map(subject_mask, dilated_mask, rgb_array)
    background_plate = reconstruct_background_rgb(rgb_array, dilated_mask)
    background_depth = complete_background_depth(refined_depth, dilated_mask)
    provenance_map = compute_provenance_map(dilated_mask)

    # 9. Save Phase C Diagnostic Artifacts
    save_phase_c_diagnostic_artifacts(hash_dir, background_plate, background_depth, provenance_map, boundary_risk_map)

    # Spatial Intelligence Analysis
    spatial_diagnostics = analyze_spatial_scene(
        rgb_array, refined_depth, background_depth, confidence_map, provenance_map,
        sel_res, hash_dir=hash_dir
    )

    # Mode Selection
    selected_mode = select_render_mode(
        requested_mode=getattr(args, "render_mode", "auto"),
        motion_style=getattr(args, "motion", "Cinematic Push-In"),
        motion_strength=getattr(args, "strength", "Cinematic")
    )

    fx, fy, cx, cy = derive_camera_intrinsics(pil_img.width, pil_img.height)
    requested_frame_count = getattr(args, "frames", 48)

    trans_plan, rot_plan, final_scale, plan_summary = plan_safe_motion_trajectory(
        args.motion, args.strength, pil_img.width, pil_img.height,
        refined_depth, confidence_map, subject_mask, boundary_risk_map, provenance_map,
        fx, fy, cx, cy, num_frames=requested_frame_count
    )

    rendered_frames = []
    per_frame_metrics = []
    output_mp4_path = None

    render_req = RenderRequest(
        input_path=input_path,
        motion_style=args.motion,
        motion_strength=args.strength,
        render_mode=selected_mode,
        quality=getattr(args, "quality", "auto"),
        resolution=getattr(args, "resolution", "auto"),
        frame_count=requested_frame_count,
        output_dir=base_output
    )

    if args.render_video:
        frames_dir = hash_dir / args.strength.lower() / "frames"

        if selected_mode == "3d":
            # Mode B: Explicit Inferred 3D Scene Pipeline
            mode_3d = Mode3DPipeline(render_req)
            m3d_res = mode_3d.execute(rgb_array, refined_depth, subject_mask)
            rendered_frames = m3d_res["frames"]
            # Save frames to disk
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i, f_img in enumerate(rendered_frames):
                Image.fromarray(f_img).save(frames_dir / f"frame_{i:04d}.png")
        else:
            # Mode A: 2.5D Parallax Pipeline
            mode_2_5d = Mode25DPipeline(render_req)
            m25_res = mode_2_5d.execute(
                rgb_array=rgb_array,
                depth_map=refined_depth,
                subject_mask=subject_mask,
                bg_plate=background_plate,
                bg_depth=background_depth,
                provenance_map=provenance_map,
                boundary_risk_map=boundary_risk_map,
                translations=trans_plan,
                rotations=rot_plan,
                fx=fx, fy=fy, cx=cx, cy=cy,
                disparity_ceiling_px=plan_summary["disparity_ceiling_target_px"],
                frames_dir=frames_dir,
                spatial_diagnostics=spatial_diagnostics,
                motion_amplitude=getattr(args, "motion_amplitude", "MEDIUM")
            )
            rendered_frames = m25_res["frames"]
            per_frame_metrics = m25_res.get("per_frame_metrics", [])

        output_mp4_path = hash_dir / args.strength.lower() / "cinematic.mp4"
        video_meta = encode_and_verify_mp4(frames_dir, output_mp4_path, fps=24, expected_frames=requested_frame_count)

    t_total = time.time() - t_start_total
    metrics_json_path = hash_dir / "metrics.json"

    result_summary = {
        "short_hash": short_hash,
        "selected_mode": selected_mode,
        "output_dir": str(hash_dir),
        "output_mp4": str(output_mp4_path) if output_mp4_path else None,
        "metrics_json": str(metrics_json_path),
        "frame_count": len(rendered_frames),
        "total_time_sec": t_total
    }

    return result_summary
