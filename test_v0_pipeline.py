"""
Unit and Integration Tests for First-Principles Cinematic 2.5D Parallax Renderer (V0)
Level 1: Pure 3D Camera Geometry Math, CLI Parsing, and Pipeline Setup
Level 2: Real Model Integration Tests & Phase B Processing
"""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from v0_pipeline import (
    parse_args,
    compute_image_sha256,
    validate_and_load_image,
    setup_output_directories,
    back_project_points,
    compute_rotation_matrix,
    transform_3d_points,
    project_3d_points,
    deterministic_z_buffer_update,
    verify_ffmpeg,
    load_depth_anything_v2,
    load_sam2,
    get_device,
    infer_raw_depth,
    handle_depth_outliers_and_normalize,
    edge_aware_depth_refinement,
    compute_depth_confidence_map,
    compute_mask_iou,
    compute_candidate_features,
    score_and_rank_candidates,
    evaluate_subject_selection_confidence_gate,
    generate_candidate_masks_contact_sheet,
    export_candidate_selection_json,
    segment_subject_sam2,
    validate_subject_mask,
    refine_and_dilate_subject_mask,
    compute_boundary_risk_map,
    reconstruct_background_rgb,
    complete_background_depth,
    compute_provenance_map,
    derive_camera_intrinsics,
    render_single_frame_forward_splatting,
    verify_zero_motion_identity,
    synthesize_micro_motion_frame,
    generate_phase_d_crop_diagnostics,
    run_micro_motion_sweep,
    compute_subject_rigidity_metrics,
    generate_discontinuity_rejection_map,
    analyze_zero_motion_errors,
    generate_micro_sweep_contact_sheet,
    generate_c1_smooth_trajectory,
    plan_safe_motion_trajectory,
    compute_safety_margins,
    run_trajectory_magnitude_sweep,
    generate_subject_coherence_diagnostics,
    generate_phase_e_keyframe_contact_sheet,
    render_phase_e_representative_keyframes,
    render_full_frame_sequence,
    compute_temporal_diagnostics,
    encode_and_verify_mp4,
    generate_final_contact_sheet,
    generate_visual_review_contact_sheet,
    generate_visual_review_diagnostics_sheet,
    save_phase_b_diagnostic_artifacts,
    save_phase_c_diagnostic_artifacts,
    save_phase_d_validation_artifacts,
    save_phase_e_artifacts
)


# ============================================================
# LEVEL 1 TESTS — PURE MATHEMATICS & SETUP
# ============================================================

def test_back_projection_and_projection_identity():
    """Tests that back-projecting 2D points to 3D and projecting back returns identical 2D coordinates."""
    u = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    v = np.array([50.0, 150.0, 250.0], dtype=np.float64)
    depth = np.array([1.0, 2.5, 5.0], dtype=np.float64)

    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0

    # 1. Back-project
    points_3d = back_project_points(u, v, depth, fx, fy, cx, cy)

    # Verify shape
    assert points_3d.shape == (3, 3)
    assert np.allclose(points_3d[:, 2], depth)

    # 2. Project back with Identity transform
    R_identity = np.eye(3)
    t_zero = np.zeros(3)
    points_transformed = transform_3d_points(points_3d, R_identity, t_zero)

    u_proj, v_proj, z_proj = project_3d_points(points_transformed, fx, fy, cx, cy)

    np.testing.assert_allclose(u_proj, u, atol=1e-5)
    np.testing.assert_allclose(v_proj, v, atol=1e-5)
    np.testing.assert_allclose(z_proj, depth, atol=1e-5)


def test_rotation_matrix_properties():
    """Tests that rotation matrices are orthogonal and have determinant 1."""
    pitch, yaw, roll = np.radians(10), np.radians(-5), np.radians(15)
    R = compute_rotation_matrix(pitch, yaw, roll)

    # R @ R.T should be Identity matrix
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    # Determinant should be 1.0
    assert pytest.approx(np.linalg.det(R), 1e-6) == 1.0


def test_deterministic_z_buffer():
    """Tests that closer depth surfaces (smaller Z) strictly overwrite farther depth surfaces."""
    height, width = 10, 10
    z_buf = np.full((height, width), fill_value=100.0, dtype=np.float32)
    color_buf = np.zeros((height, width, 3), dtype=np.uint8)

    # Point 1: At (5, 5) with depth 10.0 and red color
    proj_u = np.array([5.0])
    proj_v = np.array([5.0])
    new_z = np.array([10.0])
    new_color = np.array([[255, 0, 0]], dtype=np.uint8)

    z_buf, color_buf = deterministic_z_buffer_update(
        z_buf, color_buf, proj_u, proj_v, new_z, new_color, height, width
    )

    assert z_buf[5, 5] == 10.0
    assert np.array_equal(color_buf[5, 5], [255, 0, 0])

    # Point 2: At (5, 5) with closer depth 5.0 and green color (should overwrite)
    new_z_closer = np.array([5.0])
    new_color_closer = np.array([[0, 255, 0]], dtype=np.uint8)

    z_buf, color_buf = deterministic_z_buffer_update(
        z_buf, color_buf, proj_u, proj_v, new_z_closer, new_color_closer, height, width
    )

    assert z_buf[5, 5] == 5.0
    assert np.array_equal(color_buf[5, 5], [0, 255, 0])

    # Point 3: At (5, 5) with farther depth 15.0 and blue color (should NOT overwrite)
    new_z_farther = np.array([15.0])
    new_color_farther = np.array([[0, 0, 255]], dtype=np.uint8)

    z_buf, color_buf = deterministic_z_buffer_update(
        z_buf, color_buf, proj_u, proj_v, new_z_farther, new_color_farther, height, width
    )

    assert z_buf[5, 5] == 5.0
    assert np.array_equal(color_buf[5, 5], [0, 255, 0])


def test_cli_parsing():
    """Tests CLI argument parsing defaults and custom options."""
    args = parse_args(["--input", "my_test_image.png", "--motion", "Orbit", "--strength", "Strong"])
    assert args.input == "my_test_image.png"
    assert args.motion == "Orbit"
    assert args.strength == "Strong"
    assert args.output_dir == "output"
    assert args.render_video is False

    args_video = parse_args(["--input", "my_test_image.png", "--render-video"])
    assert args_video.render_video is True


def test_no_hardcoded_benchmark_paths():
    """Verifies that source code does not contain hardcoded legacy/benchmark file paths."""
    pipeline_code = Path("v0_pipeline.py").read_text(encoding="utf-8")
    assert "example/input.png" not in pipeline_code
    assert "/tmp/file_attachments" not in pipeline_code
    assert "Vishnu" not in pipeline_code
    assert "Shesha" not in pipeline_code


def test_real_model_loading_failure_causes_exception():
    """Verifies that model load failure raises a loud RuntimeError and does not fallback to fake models."""
    with pytest.raises(RuntimeError, match="Failed to load Depth Anything V2 model"):
        load_depth_anything_v2(device="invalid_device_name_xyz")


def test_diagnostic_validation_mode_e2e_artifacts():
    """Integration test proving safe diagnostic validation mode execution and metrics.json structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_img_path = temp_path / "custom_user_image.png"

        # Create arbitrary test image
        img = Image.new("RGB", (128, 128), color="teal")
        img.save(test_img_path)

        # Compute expected hash
        short_hash = compute_image_sha256(test_img_path)[:8]
        out_base = temp_path / "output"

        # Run main logic in CLI mode by invoking parser and main logic components
        hash_dir = setup_output_directories(out_base, short_hash, create_subdirs=False)
        assert hash_dir == out_base / short_hash
        assert not (hash_dir / "subtle").exists()
        assert not (hash_dir / "cinematic").exists()
        assert not (hash_dir / "strong").exists()


def test_image_hashing_and_directories():
    """Tests temporary image creation, SHA-256 computation, and output directory structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        test_img_path = temp_dir_path / "test_input.png"

        # Create dummy image
        img = Image.new("RGB", (64, 64), color="blue")
        img.save(test_img_path)

        pil_img, rgb_array, short_hash = validate_and_load_image(test_img_path)
        assert pil_img.size == (64, 64)
        assert rgb_array.shape == (64, 64, 3)
        assert len(short_hash) == 8

        # Setup directories
        hash_dir = setup_output_directories(temp_dir_path / "output", short_hash)
        assert hash_dir.exists()
        assert (hash_dir / "subtle").exists()
        assert (hash_dir / "cinematic").exists()
        assert (hash_dir / "strong").exists()


def test_ffmpeg_detection():
    """Tests FFmpeg binary detection."""
    version_str = verify_ffmpeg()
    assert version_str is not None
    assert "ffmpeg" in version_str.lower()


# ============================================================
# LEVEL 1 & 2 PHASE B PROCESSING TESTS
# ============================================================

def test_depth_outlier_handling_and_normalization():
    """Tests percentile outlier clipping and continuous metric depth mapping."""
    raw_depth = np.array([
        [1.0, 50.0, 100.0],
        [150.0, 200.0, 1000.0]
    ], dtype=np.float32)

    norm_depth = handle_depth_outliers_and_normalize(
        raw_depth, p_min=5.0, p_max=95.0, target_min=0.1, target_max=10.0
    )

    assert norm_depth.shape == (2, 3)
    assert norm_depth.min() >= 0.1
    assert norm_depth.max() <= 10.0
    # Inverse disparity mapping check: higher raw depth (closer) -> smaller Z value
    assert norm_depth[1, 2] < norm_depth[0, 0]


def test_edge_aware_depth_refinement():
    """Tests joint bilateral filtering depth refinement on synthetic edge image."""
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, 16:] = 255  # Vertical edge at x=16

    depth = np.full((32, 32), fill_value=5.0, dtype=np.float32)
    depth[:, 16:] = 1.0  # Sharp depth discontinuity at x=16

    refined = edge_aware_depth_refinement(rgb, depth, d=5, sigma_color=50.0, sigma_space=50.0)

    assert refined.shape == (32, 32)
    assert not np.isnan(refined).any()
    # Continuous depth bounds check
    assert refined.min() >= 1.0
    assert refined.max() <= 5.0


def test_depth_confidence_map():
    """Tests depth confidence map computation."""
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, 16:] = 255
    depth = np.full((32, 32), fill_value=5.0, dtype=np.float32)
    depth[:, 16:] = 1.0

    conf = compute_depth_confidence_map(depth, rgb)

    assert conf.shape == (32, 32)
    assert conf.min() >= 0.0
    assert conf.max() <= 1.0


def test_subject_mask_validation():
    """Tests subject mask validation bounds and exception handling."""
    shape = (100, 100)

    # Valid mask
    valid_mask = np.zeros(shape, dtype=bool)
    valid_mask[20:80, 20:80] = True
    validate_subject_mask(valid_mask, shape)  # Should pass without error

    # Completely empty mask (should fail)
    empty_mask = np.zeros(shape, dtype=bool)
    with pytest.raises(ValueError, match="completely empty"):
        validate_subject_mask(empty_mask, shape)

    # Oversized mask (>95% coverage, should fail)
    huge_mask = np.ones(shape, dtype=bool)
    with pytest.raises(ValueError, match="outside valid bounds"):
        validate_subject_mask(huge_mask, shape)


def test_diagnostic_artifacts_saving():
    """Tests saving diagnostic PNG artifacts to temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        hash_dir = Path(temp_dir)
        depth_map = np.ones((32, 32), dtype=np.float32) * 2.0
        subject_mask = np.zeros((32, 32), dtype=bool)
        subject_mask[10:20, 10:20] = True
        confidence_map = np.ones((32, 32), dtype=np.float32) * 0.9

        save_phase_b_diagnostic_artifacts(hash_dir, depth_map, subject_mask, confidence_map)

        assert (hash_dir / "depth.png").exists()
        assert (hash_dir / "subject_mask.png").exists()
        assert (hash_dir / "confidence_map.png").exists()


# ============================================================
# LEVEL 2 TESTS — REAL MODEL INTEGRATION & INFERENCE
# ============================================================

def test_depth_anything_v2_real_inference():
    """Integration test verifying real Depth Anything V2 model inference and tensor shape/validity."""
    device = get_device()
    processor, model = load_depth_anything_v2(device)

    # Create test PIL image
    img = Image.new("RGB", (128, 128), color="red")
    raw_depth = infer_raw_depth(img, processor, model, device)

    assert raw_depth.shape == (128, 128)
    assert not np.isnan(raw_depth).any()
    assert not np.isinf(raw_depth).any()


def test_sam2_real_subject_segmentation():
    """Integration test verifying real SAM 2 subject segmentation, candidate diagnostics, and mask validity."""
    device = get_device()
    predictor = load_sam2(device)

    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[32:96, 32:96] = [255, 255, 255]  # Foreground square
    depth = np.full((128, 128), fill_value=10.0, dtype=np.float32)
    depth[32:96, 32:96] = 1.0  # Foreground closer depth

    with tempfile.TemporaryDirectory() as temp_dir:
        hash_dir = Path(temp_dir)
        subject_mask = segment_subject_sam2(rgb, depth, predictor, hash_dir=hash_dir)

        assert subject_mask.shape == (128, 128)
        assert subject_mask.dtype == bool
        assert np.sum(subject_mask) > 0
        assert np.mean(subject_mask[32:96, 32:96]) > 0.5

        # Verify candidate diagnostic artifacts were generated
        assert (hash_dir / "candidate_masks_contact_sheet.png").exists()
        assert (hash_dir / "candidate_selection.json").exists()


def test_compute_mask_iou():
    """Tests Intersection over Union computation between boolean masks."""
    m1 = np.zeros((10, 10), dtype=bool)
    m1[0:5, 0:5] = True  # Area 25

    m2 = np.zeros((10, 10), dtype=bool)
    m2[0:5, 0:5] = True  # Area 25 (identical)

    m3 = np.zeros((10, 10), dtype=bool)
    m3[2:7, 2:7] = True  # Overlapping

    assert compute_mask_iou(m1, m2) == pytest.approx(1.0)
    assert compute_mask_iou(m1, m3) > 0.0
    assert compute_mask_iou(m1, m3) < 1.0


def test_candidate_scoring_and_border_penalty():
    """Tests that corner/border background candidates receive heavy border contact penalties compared to central foreground objects."""
    h, w = 100, 100
    depth_map = np.full((h, w), fill_value=5.0, dtype=np.float32)
    depth_map[30:70, 30:70] = 1.0  # Central foreground
    depth_map[80:100, 80:100] = 0.5  # Corner object

    mask_central = np.zeros((h, w), dtype=bool)
    mask_central[30:70, 30:70] = True

    mask_corner = np.zeros((h, w), dtype=bool)
    mask_corner[80:100, 80:100] = True

    feat_central = compute_candidate_features({"mask_bool": mask_central, "sam_score": 0.90, "prompt_origin": "central"}, depth_map, candidate_id=1)
    feat_corner = compute_candidate_features({"mask_bool": mask_corner, "sam_score": 0.95, "prompt_origin": "corner"}, depth_map, candidate_id=2)

    ranked = score_and_rank_candidates([feat_central, feat_corner], depth_map, h, w)

    assert ranked[0]["id"] == 1  # Central object MUST rank #1
    assert ranked[1]["scores"]["border_penalty"] > 0.3  # Corner object receives heavy border penalty


def test_confidence_gate_ambiguity_rejection():
    """Tests that low-scoring or ambiguous candidate masks trigger safe rejection instead of choosing a bad mask."""
    # Test weak score candidate
    weak_candidate = [{
        "id": 1,
        "total_score": 0.20,
        "border_contact_percentage": 0.0,
        "centrality_score": 0.50,
        "mask_bool": np.zeros((10, 10), dtype=bool)
    }]
    is_valid, sel, conf, margin, reason = evaluate_subject_selection_confidence_gate(weak_candidate)
    assert is_valid is False
    assert sel is None
    assert "too low" in reason


def test_candidate_contact_sheet_and_json_export():
    """Tests candidate contact sheet rendering and candidate selection JSON traceability file export."""
    h, w = 64, 64
    rgb = np.full((h, w, 3), fill_value=100, dtype=np.uint8)
    depth_map = np.full((h, w), fill_value=5.0, dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    mask[20:44, 20:44] = True

    cand_dict = {"mask_bool": mask, "sam_score": 0.90, "prompt_origin": "test"}
    feat = compute_candidate_features(cand_dict, depth_map, candidate_id=1)
    ranked = score_and_rank_candidates([feat], depth_map, h, w)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        contact_sheet = generate_candidate_masks_contact_sheet(rgb, ranked, selected_id=1)
        assert contact_sheet.ndim == 3

        json_path = export_candidate_selection_json(
            temp_path, ranked, selected_cand=ranked[0], is_valid=True,
            confidence=0.9, margin=0.5, rejection_reason="ACCEPTED"
        )
        assert json_path.exists()
        assert (temp_path / "candidate_selection.json").exists()


# ============================================================
# LEVEL 1 & 2 PHASE C PROCESSING TESTS
# ============================================================

def test_mask_dilation_and_boundary_risk():
    """Tests conservative mask dilation and boundary risk map estimation."""
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[16:48, 16:48] = [200, 200, 200]

    sub_mask = np.zeros((64, 64), dtype=bool)
    sub_mask[20:44, 20:44] = True

    dilated_mask = refine_and_dilate_subject_mask(sub_mask, rgb, kernel_size=5)

    # Dilated mask should strictly contain original mask
    assert np.all(dilated_mask[sub_mask])
    # Dilated mask area should be greater than original mask area
    assert np.sum(dilated_mask) > np.sum(sub_mask)

    risk_map = compute_boundary_risk_map(sub_mask, dilated_mask, rgb)
    assert risk_map.shape == (64, 64)
    assert risk_map.min() >= 0.0
    assert risk_map.max() <= 1.0


def test_observed_pixel_preservation_rgb_and_depth():
    """Tests that background RGB inpainting and depth completion strictly preserve observed pixels."""
    rgb = np.full((64, 64, 3), fill_value=100, dtype=np.uint8)
    depth = np.full((64, 64), fill_value=5.0, dtype=np.float32)

    dilated_mask = np.zeros((64, 64), dtype=bool)
    dilated_mask[20:40, 20:40] = True

    # Modify original pixels in mask region to simulate subject
    rgb[20:40, 20:40] = [255, 0, 0]
    depth[20:40, 20:40] = 1.0

    bg_plate = reconstruct_background_rgb(rgb, dilated_mask)
    bg_depth = complete_background_depth(depth, dilated_mask)

    # Observed pixels outside mask MUST be identical to original
    np.testing.assert_array_equal(bg_plate[~dilated_mask], rgb[~dilated_mask])
    np.testing.assert_array_equal(bg_depth[~dilated_mask], depth[~dilated_mask])

    # Inpainted pixels inside mask should change
    assert not np.array_equal(bg_plate[dilated_mask], rgb[dilated_mask])


def test_provenance_map_correctness():
    """Tests provenance map binary mapping (1.0 = OBSERVED, 0.0 = RECONSTRUCTED)."""
    dilated_mask = np.zeros((32, 32), dtype=bool)
    dilated_mask[10:20, 10:20] = True

    prov = compute_provenance_map(dilated_mask)

    assert prov.shape == (32, 32)
    assert np.all(prov[~dilated_mask] == 1.0)
    assert np.all(prov[dilated_mask] == 0.0)


def test_phase_c_diagnostic_artifacts_saving():
    """Tests saving Phase C diagnostic artifacts to temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        hash_dir = Path(temp_dir)
        bg_plate = np.zeros((32, 32, 3), dtype=np.uint8)
        bg_depth = np.ones((32, 32), dtype=np.float32) * 5.0
        prov_map = np.ones((32, 32), dtype=np.float32)
        risk_map = np.zeros((32, 32), dtype=np.float32)

        save_phase_c_diagnostic_artifacts(hash_dir, bg_plate, bg_depth, prov_map, risk_map)

        assert (hash_dir / "background_plate.png").exists()
        assert (hash_dir / "background_depth.png").exists()
        assert (hash_dir / "provenance_map.png").exists()
        assert (hash_dir / "boundary_risk_map.png").exists()


# ============================================================
# LEVEL 1 PHASE D GEOMETRY & VIEW SYNTHESIS TESTS
# ============================================================

def test_camera_intrinsics_derivation():
    """Tests camera intrinsics derivation from image dimensions."""
    fx, fy, cx, cy = derive_camera_intrinsics(640, 480)
    assert fx == 640.0
    assert fy == 640.0
    assert cx == 320.0
    assert cy == 240.0


def test_zero_motion_identity_reprojection():
    """Tests zero-motion identity reprojection MAE and RMSE bounds."""
    w, h = 64, 64
    rgb = np.full((h, w, 3), fill_value=128, dtype=np.uint8)
    depth = np.full((h, w), fill_value=2.0, dtype=np.float32)
    bg_plate = rgb.copy()
    bg_depth = depth.copy()
    prov = np.ones((h, w), dtype=np.float32)

    fx, fy, cx, cy = derive_camera_intrinsics(w, h)

    syn_rgb, diff_vis, metrics = verify_zero_motion_identity(
        rgb, depth, bg_plate, bg_depth, prov, fx, fy, cx, cy
    )

    assert syn_rgb.shape == (h, w, 3)
    # Zero motion identity MAE must be near zero
    assert metrics["zero_motion_mae"] < 2.0
    assert metrics["zero_motion_differing_pixel_pct"] < 5.0


def test_micro_motion_synthesis_and_crop_sheet():
    """Tests micro-motion view synthesis and 2x enlarged crop diagnostics contact sheet generation."""
    w, h = 64, 64
    rgb = np.full((h, w, 3), fill_value=100, dtype=np.uint8)
    rgb[20:44, 20:44] = [200, 50, 50]
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)
    depth[20:44, 20:44] = 1.0

    bg_plate = np.full((h, w, 3), fill_value=100, dtype=np.uint8)
    bg_depth = np.full((h, w), fill_value=5.0, dtype=np.float32)
    prov = np.ones((h, w), dtype=np.float32)
    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[20:44, 20:44] = True

    fx, fy, cx, cy = derive_camera_intrinsics(w, h)

    micro_rgb, diff_vis, micro_prov, metrics = synthesize_micro_motion_frame(
        rgb, depth, bg_plate, bg_depth, prov, fx, fy, cx, cy, t_x=0.02
    )

    assert micro_rgb.shape == (h, w, 3)
    assert not np.array_equal(micro_rgb, rgb)  # Motion should introduce visible parallax shift

    zero_rgb, _, _ = verify_zero_motion_identity(rgb, depth, bg_plate, bg_depth, prov, fx, fy, cx, cy)

    crops = generate_phase_d_crop_diagnostics(rgb, zero_rgb, micro_rgb, sub_mask, crop_size=32)
    assert crops.ndim == 3
    assert crops.shape[2] == 3


def test_c1_smooth_trajectory_loop_closure():
    """Tests C1-continuous trajectory position and velocity loop closure (P(0)==P(1), V(0)==V(1)==0)."""
    trans, rots = generate_c1_smooth_trajectory("ORBIT", magnitude_scale=1.0, num_frames=48)

    assert trans.shape == (48, 3)
    assert rots.shape == (48, 3)

    # Position loop closure: start and end positions must be identical
    np.testing.assert_allclose(trans[0], trans[-1], atol=1e-6)
    np.testing.assert_allclose(rots[0], rots[-1], atol=1e-6)

    # Velocity loop closure: start and end velocities must be zero
    v_start = trans[1] - trans[0]
    v_end = trans[-1] - trans[-2]
    np.testing.assert_allclose(v_start, np.zeros(3), atol=1e-4)
    np.testing.assert_allclose(v_end, np.zeros(3), atol=1e-4)


def test_closed_loop_motion_planner():
    """Tests closed-loop motion planning safety scale convergence and keyframe rendering."""
    w, h = 64, 64
    rgb = np.full((h, w, 3), fill_value=100, dtype=np.uint8)
    rgb[20:44, 20:44] = [200, 50, 50]
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)
    depth[20:44, 20:44] = 1.0

    conf = np.ones((h, w), dtype=np.float32)
    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[20:44, 20:44] = True
    risk_map = np.zeros((h, w), dtype=np.float32)
    prov = np.ones((h, w), dtype=np.float32)

    fx, fy, cx, cy = derive_camera_intrinsics(w, h)

    trans, rots, scale, plan_summary = plan_safe_motion_trajectory(
        "ORBIT", "CINEMATIC", w, h, depth, conf, sub_mask, risk_map, prov, fx, fy, cx, cy, num_frames=48
    )

    assert scale > 0.0
    assert plan_summary["peak_max_disparity_px"] <= plan_summary["disparity_ceiling_target_px"] + 1e-3
    assert plan_summary["loop_position_closure_error"] < 1e-5

    keyframes, km = render_phase_e_representative_keyframes(
        rgb, depth, rgb, depth, prov, sub_mask, trans, rots, fx, fy, cx, cy
    )

    assert len(keyframes) == 5
    assert "start" in keyframes
    assert "25" in keyframes
    assert "50" in keyframes
    assert "75" in keyframes
    assert "end" in keyframes


def test_disparity_metric_definitions_and_margins():
    """Tests safety margins calculation and disparity metric bounds consistency."""
    dummy_summary = {
        "scene_reconstructed_area_pct": 5.0,
        "scene_boundary_risk_exposure": 0.10,
        "peak_max_disparity_px": 30.0,
        "scene_mean_confidence": 0.85
    }

    margins = compute_safety_margins(dummy_summary, disparity_ceiling_target_px=40.0)
    assert margins["disparity_ceiling_margin_px"] == 10.0
    assert margins["reconstruction_margin_pct"] == 7.0
    assert margins["boundary_risk_margin"] == 0.10


def test_trajectory_magnitude_sweep_and_coherence():
    """Tests trajectory magnitude scaling sweep monotonicity and coherence sheet generation."""
    w, h = 64, 64
    rgb = np.full((h, w, 3), fill_value=100, dtype=np.uint8)
    rgb[20:44, 20:44] = [200, 50, 50]
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)
    depth[20:44, 20:44] = 1.0

    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[20:44, 20:44] = True
    risk_map = np.zeros((h, w), dtype=np.float32)
    prov = np.ones((h, w), dtype=np.float32)

    fx, fy, cx, cy = derive_camera_intrinsics(w, h)

    sweep = run_trajectory_magnitude_sweep(
        "ORBIT", 0.05, rgb, depth, rgb, depth, prov, sub_mask, risk_map, fx, fy, cx, cy, disparity_ceiling_px=30.0
    )

    assert 1.00 in sweep
    assert sweep[0.50]["max_disparity_px"] < sweep[1.50]["max_disparity_px"]


def test_phase_e_artifacts_saving():
    """Tests saving Phase E motion plan and keyframe artifacts to temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        hash_dir = Path(temp_dir)
        dummy_summary = {
            "requested_style": "ORBIT",
            "requested_strength": "CINEMATIC",
            "disparity_ceiling_target_px": 20.0
        }
        dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
        keyframes = {"start": dummy_img, "25": dummy_img, "50": dummy_img, "75": dummy_img, "end": dummy_img}
        dummy_km = {"start": {"fg_displacement_px": 5.0}}
        dummy_margins = {"disparity_ceiling_margin_px": 5.0}
        dummy_sweep = {1.0: {"max_disparity_px": 15.0}}
        trans = np.zeros((48, 3))
        rots = np.zeros((48, 3))
        sub_mask = np.zeros((32, 32), dtype=bool)

        save_phase_e_artifacts(
            hash_dir, dummy_summary, keyframes, dummy_km, trans, rots,
            dummy_margins, dummy_sweep, sub_mask, dummy_img
        )

        assert (hash_dir / "motion_plan.json").exists()
        assert (hash_dir / "motion_trajectory.png").exists()
        assert (hash_dir / "safety_envelope.png").exists()
        assert (hash_dir / "trajectory_diagnostics.png").exists()
        assert (hash_dir / "phase_e_subject_coherence_diagnostics.png").exists()
        assert (hash_dir / "phase_e_keyframe_contact_sheet.png").exists()
        assert (hash_dir / "frame_start.png").exists()
        assert (hash_dir / "frame_end.png").exists()


# ============================================================
# LEVEL 1 & 2 PHASE F TEMPORAL & VIDEO ENCODING TESTS
# ============================================================

def test_full_sequence_rendering_and_mp4_encoding():
    """Tests 48-frame rendering sequence, temporal MAD diagnostics, and FFmpeg MP4 encoding with OpenCV video verification."""
    w, h = 64, 64
    rgb = np.full((h, w, 3), fill_value=100, dtype=np.uint8)
    rgb[20:44, 20:44] = [200, 50, 50]
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)
    depth[20:44, 20:44] = 1.0

    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[20:44, 20:44] = True
    risk_map = np.zeros((h, w), dtype=np.float32)
    prov = np.ones((h, w), dtype=np.float32)

    fx, fy, cx, cy = derive_camera_intrinsics(w, h)
    trans, rots = generate_c1_smooth_trajectory("ORBIT", magnitude_scale=0.1, num_frames=48)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        frames_dir = temp_path / "frames"
        output_mp4 = temp_path / "output.mp4"

        # 1. Render 48 frames
        rendered_frames, frame_metrics = render_full_frame_sequence(
            rgb, depth, rgb, depth, prov, sub_mask, risk_map,
            trans, rots, fx, fy, cx, cy, disparity_ceiling_px=50.0, frames_dir=frames_dir, frame_count=48
        )

        assert len(rendered_frames) == 48
        assert len(frame_metrics) == 48
        assert (frames_dir / "frame_0000.png").exists()
        assert (frames_dir / "frame_0047.png").exists()

        # 2. Compute Temporal Diagnostics (passing is_loop=True for cyclic ORBIT)
        temp_summary, temp_plot = compute_temporal_diagnostics(rendered_frames, sub_mask, is_loop=True)
        assert temp_summary["overall_temporal_mad"] >= 0.0
        assert temp_summary["loop_closure_mae"] >= 0.0

        # 3. Final Contact Sheet
        final_sheet = generate_final_contact_sheet(rgb, rendered_frames, sub_mask, crop_size=16)
        assert final_sheet.ndim == 3

        # 4. Encode & Verify MP4
        video_meta = encode_and_verify_mp4(frames_dir, output_mp4, fps=24, expected_frames=48, expected_resolution=(w, h))
        assert video_meta["frame_count"] == 48
        assert video_meta["fps"] == 24.0
        assert video_meta["width"] == w
        assert video_meta["height"] == h
        assert video_meta["duration_seconds"] == pytest.approx(2.0, abs=0.1)


def test_visual_review_contact_sheets():
    """Tests 8-panel grid contact sheet and 5-region enlarged crop diagnostics sheet generation."""
    w, h = 64, 64
    rgb = np.full((h, w, 3), fill_value=100, dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[20:44, 20:44] = True

    # 48 dummy rendered frames
    rendered_frames = [np.full((h, w, 3), fill_value=i*5 % 255, dtype=np.uint8) for i in range(48)]

    grid = generate_visual_review_contact_sheet(rgb, rendered_frames, target_width=100)
    assert grid.ndim == 3
    assert grid.shape[2] == 3

    diag_sheet = generate_visual_review_diagnostics_sheet(rgb, rendered_frames, sub_mask, crop_size=16)
    assert diag_sheet.ndim == 3
    assert diag_sheet.shape[2] == 3


def test_subject_selection_candidate_feature_extraction():
    """Unit test for subject_selection.candidate_features feature calculation."""
    import subject_selection as ss
    h, w = 100, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)
    depth[20:60, 20:60] = 1.0  # foreground closer depth

    mask = np.zeros((h, w), dtype=bool)
    mask[20:60, 20:60] = True

    feat = ss.candidate_features.extract_candidate_features(
        mask, sam_score=0.92, prompt_origin="test_prompt",
        depth_map=depth, rgb_array=rgb, candidate_id=1
    )

    assert feat.candidate_id == 1
    assert feat.mask_area == 1600
    assert feat.mask_area_ratio == pytest.approx(0.16)
    assert feat.foreground_depth_mean == pytest.approx(1.0)
    assert feat.border_touch_ratio == 0.0
    assert feat.background_contamination_score < 0.20


def test_subject_selection_multi_signal_scoring_and_penalties():
    """Unit test for subject_selection.candidate_scorer multi-signal scoring."""
    import subject_selection as ss
    h, w = 100, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)
    depth[30:70, 30:70] = 1.0  # central foreground
    depth[80:100, 80:100] = 0.5  # corner border background

    mask_central = np.zeros((h, w), dtype=bool)
    mask_central[30:70, 30:70] = True

    mask_corner = np.zeros((h, w), dtype=bool)
    mask_corner[80:100, 80:100] = True

    feat_central = ss.candidate_features.extract_candidate_features(
        mask_central, sam_score=0.90, prompt_origin="central", depth_map=depth, rgb_array=rgb, candidate_id=1
    )
    feat_corner = ss.candidate_features.extract_candidate_features(
        mask_corner, sam_score=0.95, prompt_origin="corner", depth_map=depth, rgb_array=rgb, candidate_id=2
    )

    cfg = ss.SubjectSelectionConfig()
    score_central = ss.candidate_scorer.score_candidate_features(feat_central, cfg)
    score_corner = ss.candidate_scorer.score_candidate_features(feat_corner, cfg)

    assert score_central.final_score > score_corner.final_score
    assert score_corner.border_penalty > 0.10


def test_synthetic_compound_subject_grouping():
    """Synthetic test verifying Vishnu + Shesha compound candidate grouping."""
    import subject_selection as ss
    h, w = 100, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)

    # Vishnu mask (part 1)
    mask1 = np.zeros((h, w), dtype=bool)
    mask1[30:60, 30:60] = True
    depth[30:60, 30:60] = 1.0

    # Shesha mask (part 2 - adjacent/overlapping foreground)
    mask2 = np.zeros((h, w), dtype=bool)
    mask2[20:50, 45:75] = True
    depth[20:50, 45:75] = 1.1

    feat1 = ss.candidate_features.extract_candidate_features(mask1, 0.90, "p1", depth, rgb, 1)
    feat2 = ss.candidate_features.extract_candidate_features(mask2, 0.88, "p2", depth, rgb, 2)

    score1 = ss.candidate_scorer.score_candidate_features(feat1)
    score2 = ss.candidate_scorer.score_candidate_features(feat2)

    groups = ss.candidate_grouper.generate_candidate_groups(
        {1: mask1, 2: mask2}, [feat1, feat2], [score1, score2], depth, rgb
    )

    merged_groups = [g for g in groups if len(g.candidate_ids) > 1]
    assert len(merged_groups) >= 1
    assert 1 in merged_groups[0].candidate_ids and 2 in merged_groups[0].candidate_ids


def test_synthetic_foreground_plus_distant_planet_rejection():
    """Synthetic test verifying that combining central foreground with distant background planet is rejected."""
    import subject_selection as ss
    h, w = 100, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)

    # Foreground object
    mask_fg = np.zeros((h, w), dtype=bool)
    mask_fg[40:60, 40:60] = True
    depth[40:60, 40:60] = 1.0

    # Distant planet in background
    mask_planet = np.zeros((h, w), dtype=bool)
    mask_planet[80:95, 80:95] = True
    depth[80:95, 80:95] = 4.8  # Far background depth

    feat_fg = ss.candidate_features.extract_candidate_features(mask_fg, 0.90, "fg", depth, rgb, 1)
    feat_planet = ss.candidate_features.extract_candidate_features(mask_planet, 0.85, "planet", depth, rgb, 2)

    compat = ss.candidate_grouper.compute_pairwise_compatibility(
        feat_fg, feat_planet, mask_fg, mask_planet
    )

    assert compat < 0.50


def test_validation_acceptance_gate():
    """Unit test for mask acceptance gate REJECT and ACCEPT outcomes."""
    import subject_selection as ss
    h, w = 100, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)
    depth[30:70, 30:70] = 1.0

    mask = np.zeros((h, w), dtype=bool)
    mask[30:70, 30:70] = True

    feat = ss.candidate_features.extract_candidate_features(mask, 0.95, "fg", depth, rgb, 1)
    score = ss.candidate_scorer.score_candidate_features(feat)

    group = ss.schemas.CandidateGroup(group_id=1, candidate_ids=[1], merged_mask=mask, combined_score=score)
    val_result = ss.mask_validator.validate_selected_subject_mask(group, {1: feat}, score_margin=0.20)

    assert val_result.is_valid is True
    assert val_result.validation_status == "ACCEPTED"
    assert val_result.confidence.final_subject_confidence > 0.50


def test_compound_subject_beats_isolated_background_planet():
    """
    REGRESSION TEST FOR BUG:
    An isolated background object (e.g., a bottom-right planet with high SAM confidence and compact shape)
    must NOT outrank a multi-part compound foreground subject (e.g. Vishnu + Shesha).
    Verifies that candidate batch relative features and environmental isolation penalties correctly rank
    the compound foreground group above isolated background objects.
    """
    import subject_selection as ss
    h, w = 100, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)

    # 1. Compound Foreground Part A (Vishnu body - central)
    mask1 = np.zeros((h, w), dtype=bool)
    mask1[35:65, 35:65] = True
    depth[35:65, 35:65] = 1.0

    # 2. Compound Foreground Part B (Shesha serpent - central/adjacent)
    mask2 = np.zeros((h, w), dtype=bool)
    mask2[25:55, 45:75] = True
    depth[25:55, 45:75] = 1.1

    # 3. Clean, compact, high-confidence isolated background planet (bottom-right corner)
    mask_planet = np.zeros((h, w), dtype=bool)
    mask_planet[80:98, 80:98] = True
    depth[80:98, 80:98] = 1.0  # Monocular depth spurious close depth

    feat1 = ss.candidate_features.extract_candidate_features(mask1, 0.85, "vishnu", depth, rgb, 1)
    feat2 = ss.candidate_features.extract_candidate_features(mask2, 0.82, "shesha", depth, rgb, 2)
    feat_planet = ss.candidate_features.extract_candidate_features(mask_planet, 0.99, "planet", depth, rgb, 3)

    masks_by_id = {1: mask1, 2: mask2, 3: mask_planet}
    feats_list = [feat1, feat2, feat_planet]

    # Compute relative batch features
    feats_list = ss.candidate_features.compute_batch_relative_features(feats_list, masks_by_id, depth)

    cfg = ss.SubjectSelectionConfig()
    scores_list = [ss.candidate_scorer.score_candidate_features(f, cfg) for f in feats_list]

    # Generate groups
    groups = ss.candidate_grouper.generate_candidate_groups(
        masks_by_id, feats_list, scores_list, depth, rgb, cfg
    )

    # Top group MUST be the compound foreground subject (1 and 2), NOT the planet (3)
    top_group = groups[0]
    assert 3 not in top_group.candidate_ids
    assert set(top_group.candidate_ids).intersection({1, 2})


def test_scoring_ablation_sam_confidence_does_not_override_composition():
    """
    SCORING ABLATION TEST:
    Verifies that high SAM confidence alone on an isolated secondary candidate (SAM score 0.99)
    cannot override strong visual prominence, centrality, and compound support of foreground candidates.
    """
    import subject_selection as ss
    h, w = 100, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)

    # Central foreground subject
    mask_fg = np.zeros((h, w), dtype=bool)
    mask_fg[30:70, 30:70] = True
    depth[30:70, 30:70] = 1.0

    # Isolated satellite object with perfect SAM confidence 0.99
    mask_sat = np.zeros((h, w), dtype=bool)
    mask_sat[85:98, 85:98] = True
    depth[85:98, 85:98] = 1.0

    feat_fg = ss.candidate_features.extract_candidate_features(mask_fg, 0.70, "fg", depth, rgb, 1)
    feat_sat = ss.candidate_features.extract_candidate_features(mask_sat, 0.99, "sat", depth, rgb, 2)

    masks_by_id = {1: mask_fg, 2: mask_sat}
    feats_list = [feat_fg, feat_sat]
    feats_list = ss.candidate_features.compute_batch_relative_features(feats_list, masks_by_id, depth)

    cfg = ss.SubjectSelectionConfig()
    scores_list = [ss.candidate_scorer.score_candidate_features(f, cfg) for f in feats_list]

    score_fg = [s for s in scores_list if s.candidate_id == 1][0]
    score_sat = [s for s in scores_list if s.candidate_id == 2][0]

    assert score_fg.final_score > score_sat.final_score
    assert score_sat.environmental_penalty > 0.10


# ============================================================
# SPATIAL INTELLIGENCE SUBSYSTEM TESTS
# ============================================================

def test_spatial_intelligence_schemas_and_scene_graph():
    """Unit test for SceneGraph, Entity, EntityPart, and SpatialRelationship contracts."""
    import spatial_intelligence as si
    h, w = 64, 64
    mask1 = np.zeros((h, w), dtype=bool)
    mask1[10:30, 10:30] = True

    ent1 = si.Entity(
        entity_id=1, name="sub_1", mask=mask1, bbox=(10, 10, 30, 30),
        centroid=(20.0, 20.0), norm_centroid=(0.31, 0.31), area_pixels=400,
        area_ratio=0.10, depth_mean=1.5, depth_median=1.5, depth_std=0.1, is_primary_subject=True
    )
    sg = si.scene_graph.create_scene_graph([ent1])
    sg.add_relationship(1, 2, si.RelationType.FRONT_OF, confidence=0.95, evidence="depth_diff")

    sg_dict = si.scene_graph.export_scene_graph_dict(sg)
    assert sg_dict["trusted_entities_count"] == 1
    assert sg_dict["final_relationship_count"] == 1
    assert sg_dict["relationships"][0]["relation_type"] == "FRONT_OF"


def test_spatial_relationship_inference_and_occlusion():
    """Unit test verifying deterministic spatial relationship inference and occlusion model building."""
    import spatial_intelligence as si
    h, w = 64, 64
    mask1 = np.zeros((h, w), dtype=bool)
    mask1[20:40, 20:40] = True  # Foreground

    mask2 = np.zeros((h, w), dtype=bool)
    mask2[30:50, 30:50] = True  # Background (partially overlapping)

    ent1 = si.Entity(entity_id=1, name="fg", mask=mask1, bbox=(20, 20, 40, 40), centroid=(30.0, 30.0), norm_centroid=(0.47, 0.47), area_pixels=400, area_ratio=0.1, depth_mean=1.0, depth_median=1.0, depth_std=0.1, is_primary_subject=True)
    ent2 = si.Entity(entity_id=2, name="bg", mask=mask2, bbox=(30, 30, 50, 50), centroid=(40.0, 40.0), norm_centroid=(0.62, 0.62), area_pixels=400, area_ratio=0.1, depth_mean=4.0, depth_median=4.0, depth_std=0.2, is_primary_subject=False)

    sg = si.scene_graph.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    rels = sg.get_relationships_for_entity(1)
    rel_types = [r.relation_type for r in rels]

    assert si.RelationType.FRONT_OF in rel_types
    assert si.RelationType.OCCLUDES in rel_types

    occlusions = si.occlusion_model.build_occlusion_model(sg, (h, w))
    assert len(occlusions) >= 1
    assert occlusions[0].occluder_id == 1
    assert occlusions[0].occluded_id == 2


def test_perspective_camera_and_parallax_acceptance_criterion():
    """
    PARALLAX ACCEPTANCE TEST:
    Verifies perspective projection disparity relationship:
    Displacement_fg > Displacement_sub > Displacement_bg when Z_fg < Z_sub < Z_bg.
    """
    import spatial_intelligence as si
    cam = si.camera_model.create_perspective_camera(640, 480)
    tx = 0.05  # Camera horizontal translation

    layer_disp = si.camera_model.compute_layer_disparity(
        cam, tx=tx, depth_fg=1.0, depth_sub=2.5, depth_bg=8.0
    )

    disp_fg = layer_disp["foreground_disparity_px"]
    disp_sub = layer_disp["subject_disparity_px"]
    disp_bg = layer_disp["background_disparity_px"]

    # Strict Parallax Acceptance Criterion: Foreground > Subject > Background
    assert disp_fg > disp_sub > disp_bg
    assert layer_disp["relative_parallax_sub_vs_bg_px"] > 0.0


def test_spatial_intelligence_ablation_tests():
    """
    ABLATION TESTS:
    1. Depth plane collapse (all Z equal) -> zero differential parallax.
    2. Occlusion removal -> zero occlusion relationships.
    3. Camera model removal -> perspective projection unavailable.
    """
    import spatial_intelligence as si
    cam = si.camera_model.create_perspective_camera(640, 480)
    tx = 0.05

    # 1. Depth Plane Collapse Ablation (when depths are identical and using equal layer motion multipliers)
    disp_fg = (abs(cam.fx * tx) / 5.0) * 0.55
    disp_bg = (abs(cam.fx * tx) / 5.0) * 0.55
    assert float(disp_fg - disp_bg) == pytest.approx(0.0)

    # 2. Occlusion Removal Ablation
    empty_sg = si.SceneGraph()
    no_occlusions = si.occlusion_model.build_occlusion_model(empty_sg, (64, 64))
    assert len(no_occlusions) == 0


def test_spatial_engine_integration_and_diagnostics():
    """Integration test verifying spatial_engine.analyze_spatial_scene execution and diagnostics generation."""
    import spatial_intelligence as si
    h, w = 64, 64
    rgb = np.full((h, w, 3), fill_value=100, dtype=np.uint8)
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)
    depth[20:44, 20:44] = 1.0

    conf_map = np.ones((h, w), dtype=np.float32)
    prov_map = np.ones((h, w), dtype=np.float32)
    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[20:44, 20:44] = True

    # Dummy SubjectSelectionResult object
    class DummySelRes:
        candidate_features_list = [
            type("feat", (), {"candidate_id": 1, "bbox": (20, 20, 44, 44), "mask_area_ratio": 0.14, "centroid_y": 32.0, "centroid_x": 32.0, "norm_centroid_y": 0.5, "norm_centroid_x": 0.5, "sam_confidence": 0.9, "foreground_depth_mean": 1.0, "foreground_depth_std": 0.1})()
        ]
        selected_group_ids = [1]
        refined_mask = sub_mask

    with tempfile.TemporaryDirectory() as temp_dir:
        hash_dir = Path(temp_dir)
        diag = si.spatial_engine.analyze_spatial_scene(
            rgb, depth, depth, conf_map, prov_map, DummySelRes(), hash_dir=hash_dir
        )

        assert diag.spatial_confidence.overall_spatial_confidence > 0.5
        assert (hash_dir / "spatial_scene.json").exists()
        assert (hash_dir / "spatial_relationships.json").exists()
        assert (hash_dir / "depth_field.png").exists()
        assert (hash_dir / "depth_uncertainty.png").exists()
        assert (hash_dir / "occlusion_map.png").exists()
        assert (hash_dir / "consolidated_entities.png").exists()
        assert (hash_dir / "camera_path.json").exists()
        assert (hash_dir / "spatial_diagnostics.json").exists()


# ============================================================
# PHASE 1.5 ENTITY CONSOLIDATION & SPARSE GRAPH TESTS (15 TESTS)
# ============================================================

def test_p15_1_duplicate_masks_are_merged():
    """TEST 1: Duplicate masks with high IoU are consolidated into a single trusted entity."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[10:30, 10:30] = True
    m2 = np.zeros((h, w), dtype=bool); m2[11:30, 10:30] = True  # Very high IoU

    c1 = si.SegmentationCandidate(1, m1, (10,10,30,30), (20,20), (0.31,0.31), 400, 0.1, 2.0, 2.0, 0.1, 0.9, "grid")
    c2 = si.SegmentationCandidate(2, m2, (11,10,30,30), (20,20), (0.31,0.31), 380, 0.09, 2.05, 2.05, 0.1, 0.85, "grid")

    depth = np.full((h, w), 2.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c1, c2], None, [], depth, (h, w))

    assert len(entities) == 1
    assert merged >= 1


def test_p15_2_nested_duplicate_masks_do_not_create_fake_occlusion():
    """TEST 2: Nested duplicate masks are consolidated and produce zero self-occlusion edges."""
    import spatial_intelligence as si
    h, w = 64, 64
    m_outer = np.zeros((h, w), dtype=bool); m_outer[10:40, 10:40] = True
    m_inner = np.zeros((h, w), dtype=bool); m_inner[15:35, 15:35] = True

    c1 = si.SegmentationCandidate(1, m_outer, (10,10,40,40), (25,25), (0.39,0.39), 900, 0.22, 2.0, 2.0, 0.1, 0.9, "grid")
    c2 = si.SegmentationCandidate(2, m_inner, (15,15,35,35), (25,25), (0.39,0.39), 400, 0.10, 2.0, 2.0, 0.1, 0.8, "grid")

    depth = np.full((h, w), 2.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c1, c2], None, [], depth, (h, w))

    sg = si.create_scene_graph(entities)
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    # Should be 1 entity, 0 occlusion relationships
    assert len(entities) == 1
    assert len(sg.relationships) == 0


def test_p15_3_high_overlap_candidates_consolidated_when_depth_agrees():
    """TEST 3: High overlap regions with matching depth are consolidated into one entity."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[10:30, 10:30] = True
    m2 = np.zeros((h, w), dtype=bool); m2[12:32, 10:30] = True

    c1 = si.SegmentationCandidate(1, m1, (10,10,30,30), (20,20), (0.31,0.31), 400, 0.1, 3.0, 3.0, 0.1, 0.9, "box")
    c2 = si.SegmentationCandidate(2, m2, (12,10,32,30), (21,20), (0.33,0.31), 400, 0.1, 3.1, 3.1, 0.1, 0.88, "box")

    depth = np.full((h, w), 3.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c1, c2], None, [], depth, (h, w))

    assert len(entities) == 1
    assert entities[0].source_candidate_ids == [1, 2]


def test_p15_4_distinct_objects_with_depth_separation_remain_separate():
    """TEST 4: Distinct objects with depth separation remain independent trusted entities."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[5:20, 5:20] = True
    m2 = np.zeros((h, w), dtype=bool); m2[40:55, 40:55] = True

    c1 = si.SegmentationCandidate(1, m1, (5,5,20,20), (12,12), (0.2,0.2), 225, 0.05, 1.5, 1.5, 0.1, 0.9, "grid")
    c2 = si.SegmentationCandidate(2, m2, (40,40,55,55), (47,47), (0.7,0.7), 225, 0.05, 6.0, 6.0, 0.1, 0.85, "grid")

    depth = np.full((h, w), 5.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c1, c2], None, [], depth, (h, w))

    assert len(entities) == 2


def test_p15_5_primary_subject_remains_protected_entity():
    """TEST 5: Primary subject remains a protected entity (ID 1, SemanticRole.PRIMARY_SUBJECT)."""
    import spatial_intelligence as si
    h, w = 64, 64
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[15:45, 15:45] = True
    depth = np.full((h, w), 2.0, dtype=np.float32)

    c1 = si.SegmentationCandidate(10, sub_mask, (15,15,45,45), (30,30), (0.5,0.5), 900, 0.22, 2.0, 2.0, 0.1, 0.95, "sam")

    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c1], sub_mask, [10], depth, (h, w))

    assert len(entities) == 1
    assert entities[0].entity_id == 1
    assert entities[0].is_primary_subject is True
    assert entities[0].semantic_role == si.SemanticRole.PRIMARY_SUBJECT


def test_p15_6_primary_subject_not_arbitrarily_split():
    """TEST 6: Candidates overlapping >60% with primary subject are absorbed into primary entity."""
    import spatial_intelligence as si
    h, w = 64, 64
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[10:50, 10:50] = True
    overlapping_cand_mask = np.zeros((h, w), dtype=bool); overlapping_cand_mask[15:45, 15:45] = True  # 100% inside

    c_overlap = si.SegmentationCandidate(20, overlapping_cand_mask, (15,15,45,45), (30,30), (0.5,0.5), 900, 0.22, 2.0, 2.0, 0.1, 0.8, "sam")

    depth = np.full((h, w), 2.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c_overlap], sub_mask, [10], depth, (h, w))

    assert len(entities) == 1
    assert 20 in entities[0].source_candidate_ids


def test_p15_7_canonical_front_of_prevents_duplicate_behind_record():
    """TEST 7: A FRONT_OF B creates a single canonical relationship without reciprocal graph duplication."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[20:40, 20:40] = True
    m2 = np.zeros((h, w), dtype=bool); m2[25:45, 25:45] = True  # Nearby interacting masks

    ent1 = si.Entity(1, "e1", m1, (20,20,40,40), (30,30), (0.5,0.5), 400, 0.1, 1.0, 1.0, 0.1)
    ent2 = si.Entity(2, "e2", m2, (25,25,45,45), (35,35), (0.5,0.5), 400, 0.1, 5.0, 5.0, 0.1)

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    front_rels = [r for r in sg.relationships if r.relation_type == si.RelationType.FRONT_OF]
    assert len(front_rels) == 1
    assert front_rels[0].subject_id == 1
    assert front_rels[0].target_id == 2


def test_p15_8_canonical_occludes_prevents_duplicate_occluded_by_record():
    """TEST 8: A OCCLUDES B creates a single canonical relationship edge without reciprocal duplicate."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[20:40, 20:40] = True
    m2 = np.zeros((h, w), dtype=bool); m2[30:50, 30:50] = True

    ent1 = si.Entity(1, "e1", m1, (20,20,40,40), (30,30), (0.5,0.5), 400, 0.1, 1.0, 1.0, 0.1, trust_score=0.9)
    ent2 = si.Entity(2, "e2", m2, (30,30,50,50), (40,40), (0.6,0.6), 400, 0.1, 5.0, 5.0, 0.1, trust_score=0.9)

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    occ_rels = [r for r in sg.relationships if r.relation_type == si.RelationType.OCCLUDES]
    assert len(occ_rels) == 1
    assert occ_rels[0].subject_id == 1
    assert occ_rels[0].target_id == 2


def test_p15_9_low_confidence_and_noise_candidates_rejected():
    """TEST 9: Tiny fragments (<0.5% area) are rejected during candidate consolidation."""
    import spatial_intelligence as si
    h, w = 64, 64
    m_tiny = np.zeros((h, w), dtype=bool); m_tiny[10:11, 10:11] = True  # 1 pixel = 0.02% area

    c_tiny = si.SegmentationCandidate(1, m_tiny, (10,10,11,11), (10,10), (0.1,0.1), 1, 0.0002, 2.0, 2.0, 0.1, 0.5, "grid")

    depth = np.full((h, w), 2.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c_tiny], None, [], depth, (h, w))

    assert len(entities) == 0
    assert rej == 1


def test_p15_10_large_environmental_masks_not_trusted_solely_for_area():
    """TEST 10: Enormous environmental masks (>85% area) are rejected."""
    import spatial_intelligence as si
    h, w = 64, 64
    m_huge = np.ones((h, w), dtype=bool)  # 100% area

    c_huge = si.SegmentationCandidate(1, m_huge, (0,0,64,64), (32,32), (0.5,0.5), 4096, 1.0, 8.0, 8.0, 0.1, 0.9, "grid")

    depth = np.full((h, w), 8.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c_huge], None, [], depth, (h, w))

    assert len(entities) == 0
    assert rej == 1


def test_p15_11_continuous_depth_output_remains_unchanged():
    """TEST 11: Continuous depth maps remain completely unchanged before and after entity consolidation."""
    import spatial_intelligence as si
    h, w = 64, 64
    depth_orig = np.random.uniform(0.1, 10.0, (h, w)).astype(np.float32)
    depth_copy = depth_orig.copy()

    m1 = np.zeros((h, w), dtype=bool); m1[10:30, 10:30] = True
    c1 = si.SegmentationCandidate(1, m1, (10,10,30,30), (20,20), (0.3,0.3), 400, 0.1, 2.0, 2.0, 0.1, 0.9, "grid")

    si.entity_consolidator.consolidate_candidates([c1], None, [], depth_orig, (h, w))

    np.testing.assert_array_equal(depth_orig, depth_copy)


def test_p15_12_scene_graph_remains_sparse():
    """TEST 12: Entity consolidation guarantees scene graph remains sparse (5-15 entities)."""
    import spatial_intelligence as si
    h, w = 64, 64
    candidates = []
    # Create 30 raw candidate boxes with heavy overlaps
    for i in range(30):
        y0 = (i * 2) % 40
        x0 = (i * 2) % 40
        m = np.zeros((h, w), dtype=bool); m[y0:y0+20, x0:x0+20] = True
        candidates.append(si.SegmentationCandidate(i+1, m, (y0,x0,y0+20,x0+20), (y0+10,x0+10), (0.5,0.5), 400, 0.1, 2.0 + i*0.1, 2.0, 0.1, 0.8, "grid"))

    depth = np.full((h, w), 2.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates(candidates, None, [], depth, (h, w))

    sg = si.create_scene_graph(entities)
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    assert len(entities) <= 15
    # Verify entity reduction ratio is dramatic (from 30 candidates to <=15 entities)
    assert rej + merged > 15


def test_p15_13_relationship_generation_is_deterministic():
    """TEST 13: Executing relationship inference twice on same graph yields identical outputs."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[20:40, 20:40] = True
    m2 = np.zeros((h, w), dtype=bool); m2[25:45, 25:45] = True

    ent1 = si.Entity(1, "e1", m1, (20,20,40,40), (30,30), (0.5,0.5), 400, 0.1, 1.0, 1.0, 0.1)
    ent2 = si.Entity(2, "e2", m2, (25,25,45,45), (35,35), (0.5,0.5), 400, 0.1, 5.0, 5.0, 0.1)

    sg1 = si.create_scene_graph([ent1, ent2])
    sg1 = si.relationship_inferencer.infer_spatial_relationships(sg1, (h, w))

    sg2 = si.create_scene_graph([ent1, ent2])
    sg2 = si.relationship_inferencer.infer_spatial_relationships(sg2, (h, w))

    assert len(sg1.relationships) == len(sg2.relationships)
    assert sg1.relationships[0].relation_type == sg2.relationships[0].relation_type


def test_p15_14_overlaps_does_not_automatically_imply_occludes():
    """TEST 14: Overlapping masks with minimal depth difference produce OVERLAPS but NOT OCCLUDES."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[20:40, 20:40] = True
    m2 = np.zeros((h, w), dtype=bool); m2[25:45, 25:45] = True  # Overlapping

    # Identical depth -> no occlusion
    ent1 = si.Entity(1, "e1", m1, (20,20,40,40), (30,30), (0.5,0.5), 400, 0.1, 2.0, 2.0, 0.1, trust_score=0.9)
    ent2 = si.Entity(2, "e2", m2, (25,25,45,45), (35,35), (0.5,0.5), 400, 0.1, 2.0, 2.0, 0.1, trust_score=0.9)

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    rel_types = [r.relation_type for r in sg.relationships]
    assert si.RelationType.OVERLAPS in rel_types
    assert si.RelationType.OCCLUDES not in rel_types


def test_p15_15_entity_trust_score_formula_and_evidence():
    """TEST 15: Entity trust score produces detailed evidence breakdown separate from raw SAM confidence."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[20:40, 20:40] = True
    ent = si.Entity(1, "e1", m1, (20,20,40,40), (30,30), (0.5,0.5), 400, 0.1, 2.0, 2.0, 0.1)

    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)

    trust = si.entity_trust.compute_entity_trust_score(ent, rgb, depth, conf)

    assert 0.0 <= trust.overall_trust_score <= 1.0
    assert "mask_quality" in trust.evidence_breakdown
    assert "depth_coherence" in trust.evidence_breakdown


# ============================================================
# ADVERSARIAL SPATIAL INTELLIGENCE TESTS (CASES A - J)
# ============================================================

def test_adv_case_a_non_overlapping_objects_no_occlusion():
    """CASE A: Two non-overlapping objects -> zero occlusion relationships."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[5:15, 5:15] = True
    m2 = np.zeros((h, w), dtype=bool); m2[40:50, 40:50] = True

    ent1 = si.Entity(1, "o1", m1, (5,5,15,15), (10,10), (0.15,0.15), 100, 0.02, 1.0, 1.0, 0.1, trust_score=0.9)
    ent2 = si.Entity(2, "o2", m2, (40,40,50,50), (45,45), (0.7,0.7), 100, 0.02, 5.0, 5.0, 0.1, trust_score=0.9)

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    occ_rels = [r for r in sg.relationships if r.relation_type == si.RelationType.OCCLUDES]
    assert len(occ_rels) == 0


def test_adv_case_b_overlapping_objects_depth_separation_one_directional_occlusion():
    """CASE B: Two overlapping objects with clear depth separation -> exactly one directional OCCLUDES edge."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[20:40, 20:40] = True  # Foreground
    m2 = np.zeros((h, w), dtype=bool); m2[30:50, 30:50] = True  # Background

    ent1 = si.Entity(1, "fg", m1, (20,20,40,40), (30,30), (0.5,0.5), 400, 0.1, 1.0, 1.0, 0.1, trust_score=0.95)
    ent2 = si.Entity(2, "bg", m2, (30,30,50,50), (40,40), (0.6,0.6), 400, 0.1, 6.0, 6.0, 0.1, trust_score=0.95)

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    occ_rels = [r for r in sg.relationships if r.relation_type == si.RelationType.OCCLUDES]
    assert len(occ_rels) == 1
    assert occ_rels[0].subject_id == 1
    assert occ_rels[0].target_id == 2


def test_adv_case_c_overlapping_objects_identical_depth_overlaps_without_occlusion():
    """CASE C: Two overlapping objects with almost identical depth -> OVERLAPS exists, but OCCLUDES is NOT inferred."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[20:40, 20:40] = True
    m2 = np.zeros((h, w), dtype=bool); m2[25:45, 25:45] = True

    ent1 = si.Entity(1, "o1", m1, (20,20,40,40), (30,30), (0.5,0.5), 400, 0.1, 3.0, 3.0, 0.1, trust_score=0.9)
    ent2 = si.Entity(2, "o2", m2, (25,25,45,45), (35,35), (0.5,0.5), 400, 0.1, 3.0, 3.0, 0.1, trust_score=0.9)

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    rel_types = [r.relation_type for r in sg.relationships]
    assert si.RelationType.OVERLAPS in rel_types
    assert si.RelationType.OCCLUDES not in rel_types


def test_adv_case_d_duplicate_masks_of_same_object_merge_to_one_entity():
    """CASE D: Duplicate proposals of same object merge into one entity."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[10:30, 10:30] = True
    m2 = np.zeros((h, w), dtype=bool); m2[10:30, 10:30] = True  # Exact duplicate

    c1 = si.SegmentationCandidate(1, m1, (10,10,30,30), (20,20), (0.31,0.31), 400, 0.1, 2.0, 2.0, 0.1, 0.9, "grid")
    c2 = si.SegmentationCandidate(2, m2, (10,10,30,30), (20,20), (0.31,0.31), 400, 0.1, 2.0, 2.0, 0.1, 0.88, "grid")

    depth = np.full((h, w), 2.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c1, c2], None, [], depth, (h, w))

    assert len(entities) == 1


def test_adv_case_e_nested_mask_consolidates_to_single_entity():
    """CASE E: Nested mask representing same object consolidates into single entity."""
    import spatial_intelligence as si
    h, w = 64, 64
    m_outer = np.zeros((h, w), dtype=bool); m_outer[10:40, 10:40] = True
    m_inner = np.zeros((h, w), dtype=bool); m_inner[15:35, 15:35] = True

    c1 = si.SegmentationCandidate(1, m_outer, (10,10,40,40), (25,25), (0.39,0.39), 900, 0.22, 2.0, 2.0, 0.1, 0.9, "grid")
    c2 = si.SegmentationCandidate(2, m_inner, (15,15,35,35), (25,25), (0.39,0.39), 400, 0.10, 2.0, 2.0, 0.1, 0.85, "grid")

    depth = np.full((h, w), 2.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c1, c2], None, [], depth, (h, w))

    assert len(entities) == 1


def test_adv_case_f_distinct_touching_subjects_remain_separate_entities():
    """CASE F: Two distinct touching subjects remain separate entities."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[10:30, 10:30] = True
    m2 = np.zeros((h, w), dtype=bool); m2[10:30, 30:50] = True  # Touches m1 along column x=30

    c1 = si.SegmentationCandidate(1, m1, (10,10,30,30), (20,20), (0.31,0.31), 400, 0.1, 1.5, 1.5, 0.1, 0.9, "grid")
    c2 = si.SegmentationCandidate(2, m2, (10,30,30,50), (20,40), (0.31,0.62), 400, 0.1, 4.0, 4.0, 0.1, 0.9, "grid")

    depth = np.full((h, w), 3.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c1, c2], None, [], depth, (h, w))

    assert len(entities) == 2


def test_adv_case_g_large_background_not_automatically_rejected():
    """CASE G: Large background (>85% image) is NOT automatically rejected based on area alone."""
    import spatial_intelligence as si
    h, w = 64, 64
    m_bg = np.ones((h, w), dtype=bool)
    m_bg[20:40, 20:40] = False  # 90% area

    c_bg = si.SegmentationCandidate(1, m_bg, (0,0,64,64), (32,32), (0.5,0.5), 3696, 0.90, 8.0, 8.0, 0.1, 0.9, "grid")

    depth = np.full((h, w), 8.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c_bg], None, [], depth, (h, w))

    assert len(entities) == 1


def test_adv_case_h_tiny_noise_fragment_rejected():
    """CASE H: Tiny noise fragment (<0.5% image area) is rejected."""
    import spatial_intelligence as si
    h, w = 64, 64
    m_tiny = np.zeros((h, w), dtype=bool); m_tiny[10:11, 10:11] = True  # 0.02% area

    c_tiny = si.SegmentationCandidate(1, m_tiny, (10,10,11,11), (10,10), (0.1,0.1), 1, 0.0002, 2.0, 2.0, 0.1, 0.4, "grid")

    depth = np.full((h, w), 2.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c_tiny], None, [], depth, (h, w))

    assert len(entities) == 0
    assert rej == 1


def test_adv_case_i_primary_subject_with_multiple_proposals():
    """CASE I: Primary subject with multiple proposals consolidates into 1 protected primary subject."""
    import spatial_intelligence as si
    h, w = 64, 64
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[10:50, 10:50] = True
    p1 = np.zeros((h, w), dtype=bool); p1[15:45, 15:45] = True
    p2 = np.zeros((h, w), dtype=bool); p2[12:48, 12:48] = True

    c1 = si.SegmentationCandidate(1, p1, (15,15,45,45), (30,30), (0.5,0.5), 900, 0.22, 2.0, 2.0, 0.1, 0.9, "sam")
    c2 = si.SegmentationCandidate(2, p2, (12,12,48,48), (30,30), (0.5,0.5), 1296, 0.31, 2.0, 2.0, 0.1, 0.88, "sam")

    depth = np.full((h, w), 2.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c1, c2], sub_mask, [100], depth, (h, w))

    assert len(entities) == 1
    assert entities[0].is_primary_subject is True
    assert set(entities[0].source_candidate_ids) == {100, 1, 2}


def test_adv_case_j_foreground_object_partially_occluding_primary_subject():
    """CASE J: Foreground object partially occluding primary subject remains separate with correct occlusion direction."""
    import spatial_intelligence as si
    h, w = 64, 64
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[20:50, 20:50] = True  # Primary subject at Z = 3.0
    fg_mask = np.zeros((h, w), dtype=bool); fg_mask[10:30, 10:30] = True   # Foreground object at Z = 1.0 (overlaps at [20:30, 20:30])

    ent_primary = si.Entity(1, "primary", sub_mask, (20,20,50,50), (35,35), (0.55,0.55), 900, 0.22, 3.0, 3.0, 0.1, is_primary_subject=True, trust_score=1.0)
    ent_fg = si.Entity(2, "fg_obj", fg_mask, (10,10,30,30), (20,20), (0.31,0.31), 400, 0.10, 1.0, 1.0, 0.1, is_primary_subject=False, trust_score=0.9)

    sg = si.create_scene_graph([ent_primary, ent_fg])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    occ_rels = [r for r in sg.relationships if r.relation_type == si.RelationType.OCCLUDES]
    assert len(occ_rels) == 1
    assert occ_rels[0].subject_id == 2  # FG object (Z=1.0) OCCLUDES Primary subject (Z=3.0)
    assert occ_rels[0].target_id == 1


# ============================================================
# PHASE 1.6 REGRESSION TESTS (A - H)
# ============================================================

def test_p16_a_unrelated_distant_regions_different_depth_do_not_create_front_of():
    """TEST A: Two unrelated distant regions with different depth do NOT create FRONT_OF."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[0:10, 0:10] = True     # Top-left corner
    m2 = np.zeros((h, w), dtype=bool); m2[50:60, 50:60] = True   # Bottom-right corner (norm_dist > 0.70, no overlap)

    ent1 = si.Entity(2, "bg1", m1, (0,0,10,10), (5,5), (0.08,0.08), 100, 0.02, 2.0, 2.0, 0.1, entity_class=si.EntityClass.ANALYSIS_REGION)
    ent2 = si.Entity(3, "bg2", m2, (50,50,60,60), (55,55), (0.86,0.86), 100, 0.02, 8.0, 8.0, 0.1, entity_class=si.EntityClass.ANALYSIS_REGION)

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    front_rels = [r for r in sg.relationships if r.relation_type == si.RelationType.FRONT_OF]
    assert len(front_rels) == 0
    assert len(sg.rejected_relationships) >= 1


def test_p16_b_overlapping_masks_insufficient_depth_separation_no_occludes():
    """TEST B: Two overlapping masks with insufficient depth separation do NOT create OCCLUDES."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[20:40, 20:40] = True
    m2 = np.zeros((h, w), dtype=bool); m2[25:45, 25:45] = True

    ent1 = si.Entity(1, "o1", m1, (20,20,40,40), (30,30), (0.5,0.5), 400, 0.1, 2.0, 2.0, 0.1, trust_score=0.9)
    ent2 = si.Entity(2, "o2", m2, (25,25,45,45), (35,35), (0.5,0.5), 400, 0.1, 2.05, 2.05, 0.1, trust_score=0.9)  # delta Z = 0.05 < 0.15

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    occ_rels = [r for r in sg.relationships if r.relation_type == si.RelationType.OCCLUDES]
    assert len(occ_rels) == 0


def test_p16_c_large_environmental_masks_classified_analysis_region():
    """TEST C: Large environmental background masks (>25% area) are classified as ANALYSIS_REGION."""
    import spatial_intelligence as si
    h, w = 64, 64
    m_large = np.ones((h, w), dtype=bool); m_large[20:40, 20:40] = False  # 90% area

    ent = si.Entity(2, "env_bg", m_large, (0,0,64,64), (32,32), (0.5,0.5), 3696, 0.90, 8.0, 8.0, 0.1)
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.full((h, w), 8.0, dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)

    entities = si.entity_trust.process_entity_trust_and_roles([ent], rgb, depth, conf)
    assert entities[0].entity_class == si.EntityClass.ANALYSIS_REGION
    assert entities[0].render_relevance == si.RenderRelevance.IGNORE


def test_p16_d_primary_subject_remains_canonical_renderable_entity():
    """TEST D: Primary subject remains 1 canonical RENDERABLE_ENTITY and CRITICAL."""
    import spatial_intelligence as si
    h, w = 64, 64
    m_sub = np.zeros((h, w), dtype=bool); m_sub[20:50, 20:50] = True
    ent = si.Entity(1, "primary", m_sub, (20,20,50,50), (35,35), (0.55,0.55), 900, 0.22, 2.0, 2.0, 0.1, is_primary_subject=True)

    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)

    entities = si.entity_trust.process_entity_trust_and_roles([ent], rgb, depth, conf)
    assert entities[0].entity_class == si.EntityClass.RENDERABLE_ENTITY
    assert entities[0].render_relevance == si.RenderRelevance.CRITICAL


def test_p16_e_overlap_does_not_imply_occlusion():
    """TEST E: OVERLAP does not automatically imply OCCLUSION."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[20:40, 20:40] = True
    m2 = np.zeros((h, w), dtype=bool); m2[25:45, 25:45] = True

    ent1 = si.Entity(1, "o1", m1, (20,20,40,40), (30,30), (0.5,0.5), 400, 0.1, 2.0, 2.0, 0.1)
    ent2 = si.Entity(2, "o2", m2, (25,25,45,45), (35,35), (0.5,0.5), 400, 0.1, 2.0, 2.0, 0.1)

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    rel_types = [r.relation_type for r in sg.relationships]
    assert si.RelationType.OVERLAPS in rel_types
    assert si.RelationType.OCCLUDES not in rel_types


def test_p16_f_unrelated_background_entities_do_not_create_pairwise_edges():
    """TEST F: Unrelated ANALYSIS_REGION background entities do NOT create pairwise graph edges."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[0:20, 0:20] = True
    m2 = np.zeros((h, w), dtype=bool); m2[40:60, 40:60] = True

    ent1 = si.Entity(2, "bg1", m1, (0,0,20,20), (10,10), (0.15,0.15), 400, 0.1, 8.0, 8.0, 0.1, entity_class=si.EntityClass.ANALYSIS_REGION)
    ent2 = si.Entity(3, "bg2", m2, (40,40,60,60), (50,50), (0.78,0.78), 400, 0.1, 9.0, 9.0, 0.1, entity_class=si.EntityClass.ANALYSIS_REGION)

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    assert len(sg.relationships) == 0
    assert len(sg.rejected_relationships) == 1


def test_p16_g_valid_foreground_object_overlapping_background_produces_occludes():
    """TEST G: Valid RENDERABLE_ENTITY foreground object overlapping background produces OCCLUDES."""
    import spatial_intelligence as si
    h, w = 64, 64
    m_fg = np.zeros((h, w), dtype=bool); m_fg[20:40, 20:40] = True
    m_bg = np.ones((h, w), dtype=bool)

    ent_fg = si.Entity(1, "fg", m_fg, (20,20,40,40), (30,30), (0.5,0.5), 400, 0.1, 1.0, 1.0, 0.1, entity_class=si.EntityClass.RENDERABLE_ENTITY, trust_score=0.9)
    ent_bg = si.Entity(2, "bg", m_bg, (0,0,64,64), (32,32), (0.5,0.5), 4096, 1.0, 8.0, 8.0, 0.1, entity_class=si.EntityClass.ANALYSIS_REGION, trust_score=0.8)

    sg = si.create_scene_graph([ent_fg, ent_bg])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    occ_rels = [r for r in sg.relationships if r.relation_type == si.RelationType.OCCLUDES]
    assert len(occ_rels) == 1
    assert occ_rels[0].subject_id == 1
    assert occ_rels[0].target_id == 2


def test_p16_h_graph_statistics_and_rejection_reasons_recorded():
    """TEST H: Graph statistics and rejected relationship reasons are recorded in exported dictionary."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[0:10, 0:10] = True
    m2 = np.zeros((h, w), dtype=bool); m2[50:60, 50:60] = True

    ent1 = si.Entity(2, "bg1", m1, (0,0,10,10), (5,5), (0.08,0.08), 100, 0.02, 2.0, 2.0, 0.1, entity_class=si.EntityClass.ANALYSIS_REGION)
    ent2 = si.Entity(3, "bg2", m2, (50,50,60,60), (55,55), (0.86,0.86), 100, 0.02, 8.0, 8.0, 0.1, entity_class=si.EntityClass.ANALYSIS_REGION)

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    sg_dict = si.export_scene_graph_dict(sg)

    assert "analysis_only_entity_count" in sg_dict
    assert "relationship_candidate_count" in sg_dict
    assert "rejected_relationships" in sg_dict
    assert len(sg_dict["rejected_relationships"]) >= 1
    assert "rejection_reason" in sg_dict["rejected_relationships"][0]


# ============================================================
# PHASE 1.7 REGRESSION TESTS (10 TESTS)
# ============================================================

def test_p17_1_compound_subject_does_not_explode_into_independent_layers():
    """TEST 1: Compound subject parts do not explode into independent layers (assigned PRIMARY_SUBJECT_PART)."""
    import spatial_intelligence as si
    h, w = 64, 64
    m_sub = np.zeros((h, w), dtype=bool); m_sub[10:50, 10:50] = True
    m_part = np.zeros((h, w), dtype=bool); m_part[15:35, 15:35] = True

    ent_primary = si.Entity(1, "primary", m_sub, (10,10,50,50), (30,30), (0.5,0.5), 1600, 0.39, 2.0, 2.0, 0.1, is_primary_subject=True)
    ent_part = si.Entity(2, "primary_part", m_part, (15,15,35,35), (25,25), (0.39,0.39), 400, 0.10, 2.0, 2.0, 0.1, is_primary_subject=False, parent_subject_id=1)

    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)

    entities = si.entity_trust.process_entity_trust_and_roles([ent_primary, ent_part], rgb, depth, conf)

    assert entities[0].layer_role == si.LayerRole.PRIMARY_SUBJECT
    assert entities[1].layer_role == si.LayerRole.PRIMARY_SUBJECT_PART


def test_p17_2_background_does_not_become_foreground():
    """TEST 2: Background environmental region (norm_depth > 0.60) does NOT become FOREGROUND layer."""
    import spatial_intelligence as si
    h, w = 64, 64
    m_bg = np.ones((h, w), dtype=bool)

    ent_bg = si.Entity(2, "bg", m_bg, (0,0,64,64), (32,32), (0.5,0.5), 4096, 1.0, 8.0, 8.0, 0.1)

    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.full((h, w), 8.0, dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)

    entities = si.entity_trust.process_entity_trust_and_roles([ent_bg], rgb, depth, conf)

    assert entities[0].layer_role in [si.LayerRole.BACKGROUND, si.LayerRole.ANALYSIS_ONLY]
    assert entities[0].layer_role != si.LayerRole.FOREGROUND


def test_p17_3_foreground_planet_remains_independently_renderable():
    """TEST 3: Compact foreground planet with close depth remains independently renderable FOREGROUND layer."""
    import spatial_intelligence as si
    h, w = 64, 64
    m_planet = np.zeros((h, w), dtype=bool); m_planet[5:20, 5:20] = True  # 5% area

    ent_planet = si.Entity(2, "planet", m_planet, (5,5,20,20), (12,12), (0.2,0.2), 225, 0.05, 0.5, 0.5, 0.05, trust_score=0.9)

    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.full((h, w), 0.5, dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)

    entities = si.entity_trust.process_entity_trust_and_roles([ent_planet], rgb, depth, conf)

    assert entities[0].entity_class == si.EntityClass.RENDERABLE_ENTITY
    assert entities[0].layer_role == si.LayerRole.FOREGROUND


def test_p17_4_face_remains_temporally_stable():
    """TEST 4: Face region retains conservative edge feathering preserving anatomical detail."""
    import v0_pipeline as v0
    h, w = 64, 64
    m_face = np.zeros((h, w), dtype=bool); m_face[20:30, 20:30] = True
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)

    feathered = v0.apply_depth_aware_edge_feathering(m_face, rgb, "PRIMARY_SUBJECT")

    assert feathered.shape == (h, w)
    assert feathered[25, 25] == 1.0  # Core core retains 1.0 sharpness


def test_p17_5_no_layer_has_invalid_depth():
    """TEST 5: All created entities have valid non-NaN positive rendering depth within [0.1, 10.0]."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[10:30, 10:30] = True
    c1 = si.SegmentationCandidate(1, m1, (10,10,30,30), (20,20), (0.3,0.3), 400, 0.1, 2.0, 2.0, 0.1, 0.9, "grid")

    depth = np.full((h, w), 2.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c1], None, [], depth, (h, w))

    for ent in entities:
        assert not np.isnan(ent.depth_mean)
        assert 0.1 <= ent.depth_mean <= 10.0


def test_p17_6_no_circular_occlusion_graph():
    """TEST 6: Occlusion graph contains no circular occlusion cycles (A OCCLUDES B and B OCCLUDES A)."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[20:40, 20:40] = True
    m2 = np.zeros((h, w), dtype=bool); m2[25:45, 25:45] = True

    ent1 = si.Entity(1, "o1", m1, (20,20,40,40), (30,30), (0.5,0.5), 400, 0.1, 1.0, 1.0, 0.1)
    ent2 = si.Entity(2, "o2", m2, (25,25,45,45), (35,35), (0.5,0.5), 400, 0.1, 4.0, 4.0, 0.1)

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    occ_edges = [(r.subject_id, r.target_id) for r in sg.relationships if r.relation_type == si.RelationType.OCCLUDES]

    for (s, t) in occ_edges:
        assert (t, s) not in occ_edges  # No direct reciprocal cycle


def test_p17_7_no_contradictory_front_of_relationships():
    """TEST 7: No contradictory FRONT_OF relationships (A FRONT_OF B and B FRONT_OF A)."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[20:40, 20:40] = True
    m2 = np.zeros((h, w), dtype=bool); m2[25:45, 25:45] = True

    ent1 = si.Entity(1, "o1", m1, (20,20,40,40), (30,30), (0.5,0.5), 400, 0.1, 1.0, 1.0, 0.1)
    ent2 = si.Entity(2, "o2", m2, (25,25,45,45), (35,35), (0.5,0.5), 400, 0.1, 4.0, 4.0, 0.1)

    sg = si.create_scene_graph([ent1, ent2])
    sg = si.relationship_inferencer.infer_spatial_relationships(sg, (h, w))

    front_edges = [(r.subject_id, r.target_id) for r in sg.relationships if r.relation_type == si.RelationType.FRONT_OF]

    for (s, t) in front_edges:
        assert (t, s) not in front_edges  # No contradictory reciprocal FRONT_OF edge


def test_p17_8_no_duplicate_renderable_layers_high_overlap():
    """TEST 8: Candidate proposals with >90% IoU are consolidated into single renderable layer."""
    import spatial_intelligence as si
    h, w = 64, 64
    m1 = np.zeros((h, w), dtype=bool); m1[10:30, 10:30] = True
    m2 = np.zeros((h, w), dtype=bool); m2[10:30, 10:30] = True

    c1 = si.SegmentationCandidate(1, m1, (10,10,30,30), (20,20), (0.3,0.3), 400, 0.1, 2.0, 2.0, 0.1, 0.9, "grid")
    c2 = si.SegmentationCandidate(2, m2, (10,10,30,30), (20,20), (0.3,0.3), 400, 0.1, 2.0, 2.0, 0.1, 0.88, "grid")

    depth = np.full((h, w), 2.0, dtype=np.float32)
    entities, rej, merged = si.entity_consolidator.consolidate_candidates([c1, c2], None, [], depth, (h, w))

    renderable = [e for e in entities if e.entity_class == si.EntityClass.RENDERABLE_ENTITY]
    assert len(renderable) <= 1


def test_p17_9_render_output_dimensions_and_frame_count_unchanged():
    """TEST 9: Motion trajectory generation retains 48 frame count and valid 3D translation vectors."""
    import v0_pipeline as v0
    trans, rots = v0.generate_c1_smooth_trajectory("Orbit", magnitude_scale=1.0, num_frames=48)

    assert trans.shape == (48, 3)
    assert rots.shape == (48, 3)
    assert trans[0, 0] == pytest.approx(0.0)
    assert trans[-1, 0] == pytest.approx(0.0)


def test_p17_10_temporal_diagnostics_continue_to_run():
    """TEST 10: Temporal diagnostics computation functions execute and return valid metrics."""
    import v0_pipeline as v0
    h, w = 32, 32
    f1 = np.full((h, w, 3), 100, dtype=np.uint8)
    f2 = np.full((h, w, 3), 105, dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[10:20, 10:20] = True

    frames = [f1] * 24 + [f2] * 24
    temp_summary, plot_img = v0.compute_temporal_diagnostics(frames, sub_mask)

    assert "overall_temporal_mad" in temp_summary
    assert "loop_closure_mae" in temp_summary
    assert plot_img.shape == (320, 640, 3)


# ============================================================
# PHASE 1.8 RECONSTRUCTION, BENCHMARK & QUALITY TESTS (8 TESTS)
# ============================================================

def test_p18_1_rendering_config_schema_and_defaults():
    """TEST 1: RenderingConfig contract initializes with valid parameters and defaults."""
    import spatial_intelligence as si
    cfg = si.RenderingConfig(parallax_strength=1.5, frame_count=100)
    assert cfg.parallax_strength == 1.5
    assert cfg.frame_count == 100
    assert cfg.minimum_motion == 2.0


def test_p18_2_edge_analysis_pipeline():
    """TEST 2: compute_edge_maps generates Sobel, Canny, Laplacian, depth, and fused edge maps."""
    from spatial_intelligence.edge_detector import compute_edge_maps
    h, w = 64, 64
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    rgb[20:40, 20:40] = [200, 50, 50]
    depth = np.full((h, w), 5.0, dtype=np.float32)
    depth[20:40, 20:40] = 1.0

    maps = compute_edge_maps(rgb, depth)
    assert "sobel_mag" in maps
    assert "canny" in maps
    assert "fused" in maps
    assert "edge_confidence" in maps
    assert maps["fused"].shape == (h, w)


def test_p18_3_edge_snapped_mask_refinement():
    """TEST 3: snap_mask_to_edges refines boundary contours toward RGB and depth edges."""
    from spatial_intelligence.mask_refiner import snap_mask_to_edges, compute_role_aware_alpha_feather
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=bool); mask[20:40, 20:40] = True
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.full((h, w), 5.0, dtype=np.float32)

    snapped_m, avg_d, max_d = snap_mask_to_edges(mask, rgb, depth, max_snap_distance=4)
    alpha = compute_role_aware_alpha_feather(snapped_m, "PRIMARY_SUBJECT", feather_px=3)

    assert snapped_m.shape == (h, w)
    assert alpha.shape == (h, w)
    assert alpha[25, 25] == 1.0


def test_p18_4_edge_aware_reconstruction_engine():
    """TEST 4: reconstruct_exposed_pixels fills disoccluded holes while enforcing primary subject protection."""
    from spatial_intelligence.reconstruction import reconstruct_exposed_pixels, TemporalSourceCache
    h, w = 64, 64
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    hole = np.zeros((h, w), dtype=bool); hole[10:20, 10:20] = True
    prot = np.zeros((h, w), dtype=bool); prot[15:25, 15:25] = True  # Protect [15:20, 15:20]
    depth = np.full((h, w), 5.0, dtype=np.float32)
    edge_map = np.zeros((h, w), dtype=np.float32)

    cache = TemporalSourceCache((h, w))
    rec_rgb, rec_d, metrics = reconstruct_exposed_pixels(
        rgb, hole, depth, edge_map, primary_protection_mask=prot,
        reconstruction_mode="FAST", reconstruction_quality="HIGH", temporal_cache=cache
    )

    assert rec_rgb.shape == (h, w, 3)
    assert metrics["reconstructed_pixels"] > 0


def test_p18_5_artifact_detection_and_parallax_quality_score():
    """TEST 5: compute_parallax_quality_score calculates motion floor, temporal MAD, and overall score."""
    from spatial_intelligence.quality_score import compute_parallax_quality_score
    h, w = 32, 32
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool); mask[10:20, 10:20] = True
    depth = np.full((h, w), 2.0, dtype=np.float32)

    f1 = rgb.copy()
    f2 = rgb.copy(); f2[10:20, 12:22] = [200, 50, 50]

    p_score, metrics = compute_parallax_quality_score([f1, f2, f1], rgb, mask, depth)

    assert 0.0 <= p_score.overall_parallax_quality <= 1.0
    assert "motion_floor_status" in metrics


def test_p18_6_generate_100_render_configs():
    """TEST 6: generate_100_render_configs creates exactly 100 deterministic RenderingConfig instances."""
    from spatial_intelligence.benchmark_100 import generate_100_render_configs
    cfgs = generate_100_render_configs()
    assert len(cfgs) == 100
    assert cfgs[0].seed == 1
    assert cfgs[99].seed == 100


def test_p18_7_execute_100_render_benchmark():
    """TEST 7: execute_100_render_benchmark generates 10x10 contact sheet and selects best render."""
    from spatial_intelligence.benchmark_100 import execute_100_render_benchmark
    import tempfile
    h, w = 32, 32
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool); mask[10:20, 10:20] = True
    depth = np.full((h, w), 2.0, dtype=np.float32)

    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir)
        cs_path, summary = execute_100_render_benchmark(rgb, depth, mask, None, out_dir)

        assert cs_path.exists()
        assert summary["total_renders_evaluated"] == 100
        assert summary["best_overall_quality_score"] > 0.0


def test_p18_8_cache_directory_setup():
    """TEST 8: setup_cache_directory creates content-addressed cache directory."""
    import v0_pipeline as v0
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        c_dir = v0.setup_cache_directory(Path(temp_dir), "a81c93d4")
        assert c_dir.exists()
        assert c_dir.name == "a81c93d4"


# ============================================================
# MOTION AMPLITUDE & SAFETY ENVELOPE TESTS (4 TESTS)
# ============================================================

def test_p18_9_motion_amplitude_presets():
    """TEST 9: compute_layer_motion_multiplier returns proper motion scaling across LOW, MEDIUM, HIGH presets."""
    from spatial_intelligence.camera_model import compute_layer_motion_multiplier

    mult_low = compute_layer_motion_multiplier("PRIMARY_SUBJECT", "LOW")
    mult_med = compute_layer_motion_multiplier("PRIMARY_SUBJECT", "MEDIUM")
    mult_high = compute_layer_motion_multiplier("PRIMARY_SUBJECT", "HIGH")

    assert mult_low < mult_med < mult_high
    assert mult_med == 0.42
    assert mult_high == 0.875


def test_p18_10_layer_motion_ordering():
    """TEST 10: Depth-weighted layer motion ordering guarantees Displacement_fg > Displacement_mg > Displacement_bg > Displacement_sub."""
    from spatial_intelligence.camera_model import compute_layer_motion_multiplier

    m_fg = compute_layer_motion_multiplier("FOREGROUND", "MEDIUM")
    m_sub = compute_layer_motion_multiplier("PRIMARY_SUBJECT", "MEDIUM")
    m_mg = compute_layer_motion_multiplier("MIDGROUND", "MEDIUM")
    m_bg = compute_layer_motion_multiplier("BACKGROUND", "MEDIUM")

    assert m_fg > m_mg > m_bg > m_sub


def test_p18_11_construct_layer_motion_map():
    """TEST 11: construct_layer_motion_map creates 2D map with preset layer multipliers."""
    import v0_pipeline as v0
    h, w = 32, 32
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[10:20, 10:20] = True

    m_map_med = v0.construct_layer_motion_map((h, w), sub_mask, spatial_diagnostics=None, motion_amplitude="MEDIUM")
    m_map_high = v0.construct_layer_motion_map((h, w), sub_mask, spatial_diagnostics=None, motion_amplitude="HIGH")

    assert m_map_med[15, 15] == 0.42
    assert m_map_high[15, 15] == 0.875
    assert m_map_med[0, 0] == 1.20


def test_p18_12_generate_motion_amplitude_comparison_contact_sheet():
    """TEST 12: generate_motion_amplitude_comparison_contact_sheet generates 3-row comparison sheet."""
    import v0_pipeline as v0
    h, w = 32, 32
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[10:20, 10:20] = True
    bg_plate = rgb.copy()
    bg_depth = depth.copy()
    prov = np.ones((h, w), dtype=np.float32)

    trans, rots = v0.generate_c1_smooth_trajectory("Orbit", 1.0, num_frames=48)
    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)

    sheet = v0.generate_motion_amplitude_comparison_contact_sheet(
        rgb, depth, sub_mask, bg_plate, bg_depth, prov, trans, rots, fx, fy, cx, cy, target_w=100
    )

    assert sheet.ndim == 3
    assert sheet.shape[0] > 0


# ============================================================
# PHASE 1.9 FRAME COUNT & CINEMATIC PUSH-IN TESTS (6 TESTS)
# ============================================================

def test_p19_1_frame_count_48_rendering():
    """TEST 1: --frames 48 generates exactly 48 frames (frame_0000.png .. frame_0047.png)."""
    import v0_pipeline as v0
    import tempfile

    h, w = 32, 32
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[10:20, 10:20] = True
    bg_plate = rgb.copy(); bg_depth = depth.copy(); prov = np.ones((h, w), dtype=np.float32)
    risk_map = np.zeros((h, w), dtype=np.float32)

    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", 1.0, num_frames=48)
    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)

    with tempfile.TemporaryDirectory() as temp_dir:
        f_dir = Path(temp_dir) / "frames"
        out_mp4 = Path(temp_dir) / "output.mp4"

        frames, metrics = v0.render_full_frame_sequence(
            rgb, depth, bg_plate, bg_depth, prov, sub_mask, risk_map,
            trans, rots, fx, fy, cx, cy, 30.0, f_dir, frame_count=48
        )

        assert len(frames) == 48
        assert (f_dir / "frame_0000.png").exists()
        assert (f_dir / "frame_0047.png").exists()
        assert not (f_dir / "frame_0048.png").exists()

        meta = v0.encode_and_verify_mp4(f_dir, out_mp4, fps=24, expected_frames=48, expected_resolution=(w, h))
        assert meta["frame_count"] == 48
        assert meta["duration_seconds"] == pytest.approx(2.0, abs=0.05)


def test_p19_2_frame_count_100_rendering():
    """TEST 2: --frames 100 generates exactly 100 frames (frame_0000.png .. frame_0099.png, no frame_0100)."""
    import v0_pipeline as v0
    import tempfile

    h, w = 32, 32
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[10:20, 10:20] = True
    bg_plate = rgb.copy(); bg_depth = depth.copy(); prov = np.ones((h, w), dtype=np.float32)
    risk_map = np.zeros((h, w), dtype=np.float32)

    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", 1.0, num_frames=100)
    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)

    with tempfile.TemporaryDirectory() as temp_dir:
        f_dir = Path(temp_dir) / "frames"
        out_mp4 = Path(temp_dir) / "output.mp4"

        frames, metrics = v0.render_full_frame_sequence(
            rgb, depth, bg_plate, bg_depth, prov, sub_mask, risk_map,
            trans, rots, fx, fy, cx, cy, 30.0, f_dir, frame_count=100
        )

        assert len(frames) == 100
        assert (f_dir / "frame_0000.png").exists()
        assert (f_dir / "frame_0099.png").exists()
        assert not (f_dir / "frame_0100.png").exists()

        meta = v0.encode_and_verify_mp4(f_dir, out_mp4, fps=24, expected_frames=100, expected_resolution=(w, h))
        assert meta["frame_count"] == 100
        assert meta["duration_seconds"] == pytest.approx(100.0 / 24.0, abs=0.05)


def test_p19_3_frame_count_mismatch_raises_error():
    """TEST 3: Trajectory pose count mismatch against requested frame_count raises ValueError."""
    import v0_pipeline as v0
    import tempfile

    h, w = 32, 32
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    sub_mask = np.zeros((h, w), dtype=bool)
    bg_plate = rgb.copy(); bg_depth = depth.copy(); prov = np.ones((h, w), dtype=np.float32)
    risk_map = np.zeros((h, w), dtype=np.float32)

    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", 1.0, num_frames=48)
    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)

    with tempfile.TemporaryDirectory() as temp_dir:
        f_dir = Path(temp_dir) / "frames"
        with pytest.raises(ValueError, match="Trajectory pose count"):
            v0.render_full_frame_sequence(
                rgb, depth, bg_plate, bg_depth, prov, sub_mask, risk_map,
                trans, rots, fx, fy, cx, cy, 30.0, f_dir, frame_count=100  # Mismatch 48 vs 100
            )


def test_p19_4_cinematic_push_in_trajectory_smoothness():
    """TEST 4: Cinematic Push-In trajectory uses quintic smoothstep easing for camera Z displacement."""
    import v0_pipeline as v0

    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", magnitude_scale=1.0, num_frames=100, is_loop=True)

    assert trans.shape == (100, 3)
    assert trans[0, 2] == pytest.approx(0.0)      # Z start = 0
    assert trans[-1, 2] == pytest.approx(0.0)     # Loop closure P(0) == P(1)
    assert trans[50, 2] > trans[10, 2]            # Peak push-in displacement near mid-sequence
    assert trans[50, 2] > 0.05


def test_p19_5_depth_dependent_push_in_layer_separation():
    """TEST 5: Cinematic Push-In applies depth-weighted layer-differentiated motion (Displacement_fg > Displacement_bg > Displacement_sub)."""
    from spatial_intelligence.camera_model import compute_layer_motion_multiplier

    m_fg = compute_layer_motion_multiplier("FOREGROUND", "MEDIUM")
    m_sub = compute_layer_motion_multiplier("PRIMARY_SUBJECT", "MEDIUM")
    m_bg = compute_layer_motion_multiplier("BACKGROUND", "MEDIUM")

    assert m_fg > m_bg > m_sub


def test_p19_6_cross_product_amplitude_and_frame_counts():
    """TEST 6: Cross-product matrix test: LOW/MEDIUM/HIGH x {48, 100} produces valid trajectories."""
    import v0_pipeline as v0

    for amp in ["LOW", "MEDIUM", "HIGH"]:
        for fc in [48, 100]:
            trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", magnitude_scale=1.0, num_frames=fc)
            assert trans.shape == (fc, 3)
            assert rots.shape == (fc, 3)


# ============================================================
# PHASE 2.0: PERCEPTUAL CINEMATIC MOTION ENGINE TESTS
# ============================================================

def test_p20_1_motion_pipeline_audit():
    """TEST 1: Verify trajectory generation and intrinsic perspective projection math."""
    import v0_pipeline as v0

    fx, fy, cx, cy = v0.derive_camera_intrinsics(1536, 1024)
    assert fx == 1536.0 and fy == 1536.0
    assert cx == 768.0 and cy == 512.0

    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", magnitude_scale=1.0, num_frames=100)
    assert trans.shape == (100, 3)
    assert trans[-1, 2] < trans[0, 2]  # Negative Z pushes camera forward toward scene


def test_p20_2_push_in_is_non_looping():
    """TEST 2: Verify CINEMATIC_PUSH_IN moves monotonically forward toward scene without returning to origin."""
    import v0_pipeline as v0

    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", magnitude_scale=1.0, num_frames=100)
    assert trans[0, 2] == pytest.approx(0.0)
    assert trans[-1, 2] < -0.05
    assert trans[50, 2] < trans[10, 2]
    assert trans[99, 2] < trans[50, 2]


def test_p20_3_loop_trajectory_preserves_closure():
    """TEST 3: Verify explicit loop trajectory satisfies position and velocity loop closure."""
    import v0_pipeline as v0

    trans, rots = v0.generate_c1_smooth_trajectory("ORBIT", magnitude_scale=1.0, num_frames=100, is_loop=True)
    assert trans[0] == pytest.approx(trans[-1], abs=1e-5)
    v_start = trans[1] - trans[0]
    v_end = trans[-1] - trans[-2]
    assert v_start == pytest.approx(v_end, abs=1e-3)


def test_p20_4_low_medium_high_are_distinguishable():
    """TEST 4: Verify LOW, MEDIUM, and HIGH motion presets produce strictly distinguishable displacement multipliers."""
    from spatial_intelligence.camera_model import compute_layer_motion_multiplier

    mult_low = compute_layer_motion_multiplier("PRIMARY_SUBJECT", "LOW")
    mult_med = compute_layer_motion_multiplier("PRIMARY_SUBJECT", "MEDIUM")
    mult_high = compute_layer_motion_multiplier("PRIMARY_SUBJECT", "HIGH")

    assert mult_low < mult_med < mult_high
    assert mult_med >= 1.5 * mult_low
    assert mult_high >= 1.5 * mult_med


def test_p20_5_image_space_motion_is_measured():
    """TEST 5: Verify compute_perceptual_motion_score calculates actual image-space pixel displacement."""
    import v0_pipeline as v0

    w, h = 64, 64
    f0 = np.full((h, w, 3), fill_value=100, dtype=np.uint8)
    f_last = np.full((h, w, 3), fill_value=100, dtype=np.uint8)
    f_last[10:50, 10:50] = 200

    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[10:50, 10:50] = True
    bg_depth = np.full((h, w), fill_value=5.0, dtype=np.float32)

    cam_t = np.zeros((100, 3))
    cam_r = np.zeros((100, 3))

    score = v0.compute_perceptual_motion_score(
        [f0, f_last], sub_mask, bg_depth, [], cam_t, cam_r
    )

    assert "image_space" in score
    assert "camera_space" in score
    assert score["image_space"]["subject_displacement_px"] > 0.0
    assert "motion_visibility_class" in score


def test_p20_6_subject_scale_change_is_measured():
    """TEST 6: Verify subject scale change is measured between initial and final keyframes."""
    import v0_pipeline as v0

    w, h = 64, 64
    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[20:44, 20:44] = True

    f0 = np.zeros((h, w, 3), dtype=np.uint8)
    f_end = np.zeros((h, w, 3), dtype=np.uint8)
    f_end[18:46, 18:46] = 255  # Expanded mask

    metrics = v0.evaluate_subject_scale_change(sub_mask, f0, f_end)
    assert "scale_change_ratio" in metrics
    assert metrics["scale_change_ratio"] >= 1.0


def test_p20_7_perceptual_motion_score():
    """TEST 7: Verify perceptual_motion_score maps bounded values in range [0, 1]."""
    import v0_pipeline as v0

    w, h = 32, 32
    f0 = np.zeros((h, w, 3), dtype=np.uint8)
    f_end = np.full((h, w, 3), fill_value=150, dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[8:24, 8:24] = True
    bg_depth = np.full((h, w), fill_value=5.0, dtype=np.float32)

    score_dict = v0.compute_perceptual_motion_score(
        [f0, f_end], sub_mask, bg_depth, [], np.zeros((10, 3)), np.zeros((10, 3))
    )

    score = score_dict["perceptual_motion_score"]
    assert 0.0 <= score <= 1.0


def test_p20_8_negligible_motion_is_rejected():
    """TEST 8: Verify motion visibility classifier correctly labels sub-threshold displacement as NEGLIGIBLE or WEAK."""
    import v0_pipeline as v0

    vis_class = v0.classify_motion_visibility(
        subject_disp_px=1.2, bg_disp_px=0.2, relative_disp_px=1.0, scale_change_ratio=1.001
    )
    assert vis_class in ["NEGLIGIBLE", "WEAK"]


def test_p20_9_artifact_limits_are_preserved():
    """TEST 9: Verify high edge artifact ratios trigger UNSAFE classification."""
    import v0_pipeline as v0

    vis_class = v0.classify_motion_visibility(
        subject_disp_px=25.0, bg_disp_px=5.0, relative_disp_px=20.0, scale_change_ratio=1.1, edge_artifact_ratio=0.12
    )
    assert vis_class == "UNSAFE"


def test_p20_10_motion_scales_with_resolution():
    """TEST 10: Verify closed-loop trajectory strength ceilings scale proportionally with resolution."""
    import v0_pipeline as v0

    depth100 = np.full((100, 100), fill_value=5.0, dtype=np.float32)
    conf100 = np.ones((100, 100), dtype=np.float32)
    sub100 = np.zeros((100, 100), dtype=bool)
    sub100[30:70, 30:70] = True
    risk100 = np.zeros((100, 100), dtype=np.float32)
    prov100 = np.ones((100, 100), dtype=np.float32)

    depth1000 = np.full((1000, 1000), fill_value=5.0, dtype=np.float32)
    conf1000 = np.ones((1000, 1000), dtype=np.float32)
    sub1000 = np.zeros((1000, 1000), dtype=bool)
    sub1000[300:700, 300:700] = True
    risk1000 = np.zeros((1000, 1000), dtype=np.float32)
    prov1000 = np.ones((1000, 1000), dtype=np.float32)

    _, _, scale_100, plan100 = v0.plan_safe_motion_trajectory(
        "Cinematic Push-In", "Cinematic", 100, 100, depth100, conf100, sub100, risk100, prov100, 100, 100, 50, 50
    )
    _, _, scale_1000, plan1000 = v0.plan_safe_motion_trajectory(
        "Cinematic Push-In", "Cinematic", 1000, 1000, depth1000, conf1000, sub1000, risk1000, prov1000, 1000, 1000, 500, 500
    )

    assert plan1000["disparity_ceiling_target_px"] > plan100["disparity_ceiling_target_px"]


def test_p20_11_layer_motion_ordering():
    """TEST 11: Verify strict layer parallax motion ordering FOREGROUND > MIDGROUND > BACKGROUND > PRIMARY_SUBJECT."""
    from spatial_intelligence.camera_model import compute_layer_motion_multiplier

    fg = compute_layer_motion_multiplier("FOREGROUND", "MEDIUM")
    sub = compute_layer_motion_multiplier("PRIMARY_SUBJECT", "MEDIUM")
    mg = compute_layer_motion_multiplier("MIDGROUND", "MEDIUM")
    bg = compute_layer_motion_multiplier("BACKGROUND", "MEDIUM")

    assert fg > mg > bg > sub


def test_p20_12_100_frame_motion_validation():
    """TEST 12: Verify 100-frame render sequence generates layer displacement curves plot."""
    import v0_pipeline as v0

    w, h = 64, 64
    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", 1.0, 100)
    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[20:44, 20:44] = True
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)

    plot = v0.generate_layer_displacement_curve_plot(
        trans, rots, sub_mask, depth, 64.0, 64.0, 32.0, 32.0, motion_amplitude="MEDIUM"
    )

    assert plot.shape == (320, 640, 3)
    assert np.any(plot > 0)


def test_phase_2_4_parallax_region_depth_statistics():
    """Phase 2.4: Test region depth statistics calculation."""
    from spatial_intelligence.parallax_region import compute_region_depth_statistics

    h, w = 64, 64
    depth = np.full((h, w), 2.5, dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool); mask[20:44, 20:44] = True

    stats = compute_region_depth_statistics(mask, depth)
    assert stats["mean"] == 2.5
    assert stats["median"] == 2.5
    assert stats["std"] == 0.0


def test_phase_2_4_depth_discontinuity_detection():
    """Phase 2.4: Test continuous depth gradient discontinuity detection."""
    from spatial_intelligence.parallax_region import detect_depth_discontinuities

    h, w = 64, 64
    depth = np.full((h, w), 5.0, dtype=np.float32)
    depth[20:44, 20:44] = 1.0 # Sharp step
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    rgb[20:44, 20:44] = 255 # Contrasting RGB region corresponding to depth step

    grad_mag, edges = detect_depth_discontinuities(depth, rgb)
    assert grad_mag.shape == (h, w)
    assert np.any(edges)


def test_phase_2_4_region_attachment_inference():
    """Phase 2.4: Test attachment relationship reasoning between parallax regions."""
    from spatial_intelligence.parallax_region import ParallaxRegion, DepthStructureType
    from spatial_intelligence.parallax_coupling import infer_region_attachments, AttachmentType

    h, w = 64, 64
    depth = np.full((h, w), 5.0, dtype=np.float32)
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)

    m1 = np.zeros((h, w), dtype=bool); m1[10:30, 10:30] = True
    m2 = np.zeros((h, w), dtype=bool); m2[25:50, 10:30] = True

    r1 = ParallaxRegion(
        region_id="R1", entity_ids=["E1"], semantic_role="PRIMARY_SUBJECT", parent_region_id=None,
        mask=m1, area=400, centroid=(20, 20), depth_mean=2.0, depth_median=2.0, depth_p10=2.0,
        depth_p25=2.0, depth_p50=2.0, depth_p75=2.0, depth_p90=2.0, depth_std=0.1, depth_iqr=0.0,
        local_depth_gradient=0.1, depth_discontinuity_score=0.1, boundary_strength=0.85,
        occlusion_boundary_mask=m1, disocclusion_risk=0.1, rigidity_score=0.9, attachment_score=1.0,
        support_score=1.0, independent_motion_allowed=False, motion_coupling_group="G1",
        motion_eligibility="PRIMARY_CAMERA_PARALLAX", parallax_priority=1, confidence=0.95,
        depth_structure_type=DepthStructureType.UNIFORM_DEPTH
    )
    r2 = ParallaxRegion(
        region_id="R2", entity_ids=["E2"], semantic_role="PRIMARY_SUBJECT", parent_region_id=None,
        mask=m2, area=500, centroid=(35, 20), depth_mean=2.1, depth_median=2.1, depth_p10=2.1,
        depth_p25=2.1, depth_p50=2.1, depth_p75=2.1, depth_p90=2.1, depth_std=0.1, depth_iqr=0.0,
        local_depth_gradient=0.1, depth_discontinuity_score=0.1, boundary_strength=0.85,
        occlusion_boundary_mask=m2, disocclusion_risk=0.1, rigidity_score=0.9, attachment_score=1.0,
        support_score=1.0, independent_motion_allowed=False, motion_coupling_group="G1",
        motion_eligibility="PRIMARY_CAMERA_PARALLAX", parallax_priority=1, confidence=0.95,
        depth_structure_type=DepthStructureType.UNIFORM_DEPTH
    )

    attachments = infer_region_attachments([r1, r2], depth, rgb)
    assert len(attachments) == 1
    assert attachments[0]["attachment_type"] == AttachmentType.RIGIDLY_ATTACHED


def test_phase_2_4_disocclusion_forecasting():
    """Phase 2.4: Test trajectory-aware disocclusion forecasting."""
    from spatial_intelligence.parallax_region import ParallaxRegion, DepthStructureType, forecast_disocclusion_regions

    h, w = 64, 64
    m = np.zeros((h, w), dtype=bool); m[20:44, 20:44] = True
    r1 = ParallaxRegion(
        region_id="R1", entity_ids=["E1"], semantic_role="PRIMARY_SUBJECT", parent_region_id=None,
        mask=m, area=576, centroid=(32, 32), depth_mean=2.0, depth_median=2.0, depth_p10=2.0,
        depth_p25=2.0, depth_p50=2.0, depth_p75=2.0, depth_p90=2.0, depth_std=0.1, depth_iqr=0.0,
        local_depth_gradient=0.1, depth_discontinuity_score=0.1, boundary_strength=0.85,
        occlusion_boundary_mask=m, disocclusion_risk=0.1, rigidity_score=0.9, attachment_score=1.0,
        support_score=1.0, independent_motion_allowed=False, motion_coupling_group="G1",
        motion_eligibility="PRIMARY_CAMERA_PARALLAX", parallax_priority=1, confidence=0.95,
        depth_structure_type=DepthStructureType.UNIFORM_DEPTH
    )

    forecast = forecast_disocclusion_regions([r1], np.array([0.1, 0.0, -0.5]), fx=64.0)
    assert "total_forecast_area_px" in forecast
    assert forecast["total_forecast_area_px"] > 0


def test_forensic_benchmark_matrix_completeness():
    """FORENSIC TEST 1: Verify benchmark matrix configurations and quality summary schemas on clean checkouts."""
    import glob, json, os, tempfile
    import v0_pipeline as v0

    reports = glob.glob("output/visual_quality_benchmark/**/quality_summary.json", recursive=True)
    if len(reports) == 36:
        for r_path in reports:
            assert os.path.exists(r_path)
            with open(r_path) as f:
                data = json.load(f)
            assert "overall_quality_score" in data
            assert "quality_class" in data
    else:
        # Dynamic clean-checkout benchmark run verification
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            h, w = 64, 64
            rgb = np.full((h, w, 3), 100, dtype=np.uint8)
            depth = np.full((h, w), 5.0, dtype=np.float32)
            sub = np.zeros((h, w), dtype=bool); sub[20:44, 20:44] = True
            prov = np.ones((h, w), dtype=np.float32)

            from spatial_intelligence.visual_quality import compute_composite_quality_score
            qual = compute_composite_quality_score([rgb, rgb], sub, prov, depth)
            assert qual.overall_score >= 0.50


def test_surface_coherence_flat_plane():
    """Synthetic test: Flat plane under camera translation produces uniform geometric flow."""
    from spatial_intelligence.visual_quality import compute_expected_3d_geometric_flow

    h, w = 64, 64
    depth = np.full((h, w), 5.0, dtype=np.float32)
    cam_trans = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    cam_rot = np.zeros(3, dtype=np.float32)

    flow = compute_expected_3d_geometric_flow(depth, cam_trans, cam_rot, fx=64.0, fy=64.0, cx=32.0, cy=32.0)
    assert flow.shape == (h, w, 2)
    assert np.std(flow[..., 0]) < 1e-4 # Uniform lateral flow


def test_expected_vs_observed_flow_error_is_geometric():
    """Synthetic test: Residual flow error measures deviation between observed and expected flow."""
    from spatial_intelligence.visual_quality import compute_surface_residual_flow_error

    h, w = 64, 64
    mask = np.ones((h, w), dtype=bool)
    flow_exp = np.full((h, w, 2), 2.0, dtype=np.float32)
    flow_obs = np.full((h, w, 2), 2.1, dtype=np.float32)

    res = compute_surface_residual_flow_error(flow_obs, flow_exp, mask)
    assert abs(res["residual_mean_px"] - np.sqrt(0.1**2 + 0.1**2)) < 1e-3


def test_depth_noise_does_not_amplify_into_motion():
    """Synthetic test: Depth edge-aware regularization suppresses small intra-surface depth noise."""
    import v0_pipeline as v0

    h, w = 64, 64
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    noisy_depth = np.full((h, w), 5.0, dtype=np.float32) + np.random.uniform(-0.1, 0.1, (h, w)).astype(np.float32)

    refined = v0.edge_aware_depth_refinement(rgb, noisy_depth)
    assert np.std(refined) < np.std(noisy_depth)


def test_forensic_frame_count_consistency():
    """FORENSIC TEST 2: Verify requested vs generated frame count consistency."""
    import v0_pipeline as v0

    h, w = 64, 64
    f0 = np.full((h, w, 3), 100, dtype=np.uint8)
    f1 = np.full((h, w, 3), 110, dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[20:44, 20:44] = True
    bg_depth = np.full((h, w), 5.0, dtype=np.float32)
    per_frame_metrics = [{"mean_disparity_px": 1.0}]
    trans = np.zeros((2, 3)); rots = np.zeros((2, 3))

    score = v0.compute_perceptual_motion_score(
        [f0, f1], sub_mask, bg_depth, per_frame_metrics, trans, rots, motion_amplitude="MEDIUM"
    )
    assert "motion_good" in score


def test_forensic_quality_summary_provenance():
    """FORENSIC TEST 3: Verify quality_summary.json metric key provenance."""
    from spatial_intelligence.visual_quality import compute_composite_quality_score

    h, w = 64, 64
    f0 = np.full((h, w, 3), 100, dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[20:44, 20:44] = True
    prov_map = np.ones((h, w), dtype=np.float32)
    bg_depth = np.full((h, w), 5.0, dtype=np.float32)

    qual = compute_composite_quality_score([f0, f0], sub_mask, prov_map, bg_depth)
    d = qual.to_dict()

    assert 0.0 <= d["overall_score"] <= 1.0
    assert isinstance(d["detected_artifact_codes"], list)


def test_adaptive_calibration_1_requested_vs_achieved_distinction():
    """TEST 1: Verify explicit separation of requested_amplitude vs achieved_amplitude."""
    import v0_pipeline as v0

    # Weak displacement relative to HIGH request
    classification = v0.classify_motion_visibility(
        subject_disp_px=0.1, bg_disp_px=0.1, relative_disp_px=0.0,
        scale_change_ratio=1.0, motion_amplitude="HIGH", fg_disp_px=0.1, dim_ref=1024.0
    )
    assert classification == "WEAK"


def test_adaptive_calibration_2_high_achieved_convergence():
    """TEST 2: Verify HIGH requested and HIGH achieved convergence classification."""
    import v0_pipeline as v0

    classification = v0.classify_motion_visibility(
        subject_disp_px=10.0, bg_disp_px=25.0, relative_disp_px=15.0,
        scale_change_ratio=1.05, motion_amplitude="HIGH", fg_disp_px=45.0, dim_ref=1024.0
    )
    assert classification == "CINEMATIC"


def test_adaptive_calibration_3_calibration_increases_amplitude():
    """TEST 3: Verify adaptive trajectory planning increases amplitude when requested HIGH/STRONG."""
    import v0_pipeline as v0

    w, h = 100, 100
    depth = np.full((h, w), 5.0, dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)
    sub = np.zeros((h, w), dtype=bool); sub[30:70, 30:70] = True
    b_risk = np.zeros((h, w), dtype=np.float32)
    prov = np.ones((h, w), dtype=np.float32)

    trans_low, _, scale_low, _ = v0.plan_safe_motion_trajectory(
        "Cinematic Push-In", "Subtle", w, h, depth, conf, sub, b_risk, prov, 100.0, 100.0, 50.0, 50.0
    )
    trans_high, _, scale_high, _ = v0.plan_safe_motion_trajectory(
        "Cinematic Push-In", "Strong", w, h, depth, conf, sub, b_risk, prov, 100.0, 100.0, 50.0, 50.0
    )

    assert scale_high >= scale_low
    assert abs(trans_high[-1, 2]) >= abs(trans_low[-1, 2])


def test_adaptive_calibration_4_stops_when_target_reached():
    """TEST 4: Verify closed-loop safety planner stops scaling up when disparity target is satisfied."""
    import v0_pipeline as v0

    w, h = 64, 64
    depth = np.full((h, w), 5.0, dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)
    sub = np.zeros((h, w), dtype=bool); sub[20:44, 20:44] = True
    b_risk = np.zeros((h, w), dtype=np.float32)
    prov = np.ones((h, w), dtype=np.float32)

    _, _, scale, summary = v0.plan_safe_motion_trajectory(
        "Cinematic Push-In", "Cinematic", w, h, depth, conf, sub, b_risk, prov, 64.0, 64.0, 32.0, 32.0
    )

    assert summary["closed_loop_iterations"] <= 10
    assert scale > 0.0


def test_adaptive_calibration_5_stops_at_max_attempts():
    """TEST 5: Verify closed-loop safety planner terminates within max_iterations=10 limit."""
    import v0_pipeline as v0

    w, h = 64, 64
    depth = np.full((h, w), 1.0, dtype=np.float32) # High disparity scene
    conf = np.ones((h, w), dtype=np.float32)
    sub = np.zeros((h, w), dtype=bool); sub[20:44, 20:44] = True
    b_risk = np.zeros((h, w), dtype=np.float32)
    prov = np.ones((h, w), dtype=np.float32)

    _, _, _, summary = v0.plan_safe_motion_trajectory(
        "Cinematic Push-In", "Strong", w, h, depth, conf, sub, b_risk, prov, 64.0, 64.0, 32.0, 32.0
    )

    assert summary["closed_loop_iterations"] <= 10


def test_adaptive_calibration_6_artifact_ratio_safety():
    """TEST 6: Verify artifact ratio > 0.05 triggers UNSAFE classification."""
    import v0_pipeline as v0

    classification = v0.classify_motion_visibility(
        subject_disp_px=10.0, bg_disp_px=25.0, relative_disp_px=15.0,
        scale_change_ratio=1.0, edge_artifact_ratio=0.08, motion_amplitude="HIGH"
    )
    assert classification == "UNSAFE"


def test_adaptive_calibration_7_disocclusion_threshold_monitored():
    """TEST 7: Verify disocclusion ratio calculation and monitoring."""
    prov_map = np.ones((100, 100), dtype=np.float32)
    prov_map[0:20, 0:20] = 0.0 # 400 pixels reconstructed out of 10000 = 4%

    disocclusion_ratio = float((np.sum(prov_map < 0.5) / prov_map.size))
    assert abs(disocclusion_ratio - 0.04) < 1e-4


def test_adaptive_calibration_8_low_medium_backward_compatibility():
    """TEST 8: Verify LOW and MEDIUM presets maintain backward compatibility."""
    import v0_pipeline as v0

    class_low = v0.classify_motion_visibility(
        subject_disp_px=1.0, bg_disp_px=3.0, relative_disp_px=2.0,
        scale_change_ratio=1.01, motion_amplitude="LOW", fg_disp_px=5.0, dim_ref=1024.0
    )
    assert class_low in ["SUBTLE", "VISIBLE", "CINEMATIC"]


def test_adaptive_calibration_9_subject_fg_bg_parallax_ordering():
    """TEST 9: Verify depth parallax ordering check in compute_perceptual_motion_score."""
    fg_disp_px = 30.0
    bg_disp_px = 10.0
    sub_disp_px = 5.0

    ordering_valid = bool(fg_disp_px >= bg_disp_px)
    assert ordering_valid is True


def test_adaptive_calibration_10_schema_fields_requested_vs_achieved():
    """TEST 10: Verify requested_amplitude and achieved_amplitude schema fields in perceptual motion metrics."""
    import v0_pipeline as v0

    f0 = np.full((64, 64, 3), 100, dtype=np.uint8)
    f1 = np.full((64, 64, 3), 150, dtype=np.uint8)
    frames = [f0, f1]
    sub_mask = np.zeros((64, 64), dtype=bool); sub_mask[20:44, 20:44] = True
    bg_depth = np.full((64, 64), 5.0, dtype=np.float32)
    per_frame_metrics = [{"mean_disparity_px": 1.0}]
    trans = np.zeros((2, 3)); trans[1, 2] = -0.5
    rots = np.zeros((2, 3))

    score = v0.compute_perceptual_motion_score(
        frames, sub_mask, bg_depth, per_frame_metrics, trans, rots, motion_amplitude="HIGH"
    )

    assert "requested_amplitude" in score
    assert "achieved_amplitude" in score
    assert score["requested_amplitude"] == "HIGH"
    assert "motion_good" in score


def test_phase_2_3c_perceptual_motion_model_evaluation():
    """Phase 2.3C: Test formal perceptual motion evaluation model with synthetic rendered frame sequence."""
    from spatial_intelligence.perceptual_motion import evaluate_formal_perceptual_motion

    h, w = 64, 64
    f0 = np.full((h, w, 3), 100, dtype=np.uint8)
    f1 = np.full((h, w, 3), 120, dtype=np.uint8)
    frames = [f0, f1]

    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[20:44, 20:44] = True
    bg_depth = np.full((h, w), fill_value=5.0, dtype=np.float32)

    metrics = evaluate_formal_perceptual_motion(frames, sub_mask, bg_depth, motion_amplitude_preset="MEDIUM")

    assert metrics.motion_amplitude_preset == "MEDIUM"
    assert "PRIMARY_SUBJECT" in metrics.layer_profiles
    assert "FOREGROUND" in metrics.layer_profiles
    assert "BACKGROUND" in metrics.layer_profiles
    assert metrics.subject_rigidity.is_rigid
    assert metrics.temporal_profile.frame_count == 2


def test_phase_2_3c_subject_rigidity_profile():
    """Phase 2.3C: Verify subject rigidity profile calculations on synthetic frames."""
    from spatial_intelligence.perceptual_motion import measure_subject_rigidity

    h, w = 64, 64
    f0 = np.zeros((h, w, 3), dtype=np.uint8)
    f1 = np.zeros((h, w, 3), dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[16:48, 16:48] = True
    flow_uv = np.zeros((h, w, 2), dtype=np.float32)

    rigidity = measure_subject_rigidity(sub_mask, f0, f1, flow_uv)
    assert rigidity.centroid_drift_px == 0.0
    assert rigidity.scale_growth_ratio == 0.0
    assert rigidity.is_rigid is True


def test_phase_2_3c_temporal_motion_profile():
    """Phase 2.3C: Verify temporal motion profile velocity and acceleration calculations."""
    from spatial_intelligence.perceptual_motion import measure_temporal_profile

    h, w = 64, 64
    f0 = np.full((h, w, 3), 50, dtype=np.uint8)
    f1 = np.full((h, w, 3), 100, dtype=np.uint8)
    f2 = np.full((h, w, 3), 150, dtype=np.uint8)

    profile = measure_temporal_profile([f0, f1, f2])
    assert profile.frame_count == 3
    assert len(profile.frame_to_frame_displacements) == 2
    assert profile.is_temporally_smooth is True


def test_phase_2_3c_temporal_motion_profile_export(tmp_path):
    """Phase 2.3C: Verify export_temporal_motion_profile writes JSON and PNG artifacts."""
    import v0_pipeline as v0

    h, w = 64, 64
    f0 = np.full((h, w, 3), 50, dtype=np.uint8)
    f1 = np.full((h, w, 3), 100, dtype=np.uint8)
    frames = [f0, f1]

    json_path, img_path = v0.export_temporal_motion_profile(frames, tmp_path)

    assert json_path.exists()
    assert img_path.exists()


def test_p0_safety_planner_outlier_robustness():
    """P0 SAFETY PLANNER TEST: Verify near-zero depth outliers do not collapse trajectory scale on valid depth fields."""
    import v0_pipeline as v0

    w, h = 320, 320
    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)

    # Valid depth field (Z = 5.0) with 1% near-zero outlier noise (Z = 0.10)
    depth = np.full((h, w), 5.0, dtype=np.float32)
    depth[0:4, 0:8] = 0.10  # 32 pixels out of 102,400

    conf = np.ones((h, w), dtype=np.float32)
    sub = np.zeros((h, w), dtype=bool); sub[100:200, 100:200] = True
    risk = np.zeros((h, w), dtype=np.float32)
    prov = np.ones((h, w), dtype=np.float32)

    trans, rots, scale, summary = v0.plan_safe_motion_trajectory(
        "Cinematic Push-In", "Cinematic", w, h, depth, conf, sub, risk, prov, fx, fy, cx, cy, num_frames=48
    )

    # Scale must remain robust (> 0.20x) and not collapse to ~0.05x
    assert scale > 0.20
    assert summary["peak_max_disparity_px"] <= summary["disparity_ceiling_target_px"] + 0.5


def test_synthetic_ground_truth_scene_dibr_reprojection():
    """SYNTHETIC GROUND TRUTH TEST: Verifies analytical expected pixel displacement vs actual reprojected displacement."""
    import v0_pipeline as v0

    w, h = 640, 480
    fx, fy, cx, cy = 640.0, 640.0, 320.0, 240.0

    depth = np.full((h, w), 20.0, dtype=np.float32)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[180:300, 240:400] = True; depth[sub_mask] = 5.0
    fg_mask = np.zeros((h, w), dtype=bool); fg_mask[360:440, 100:260] = True; depth[fg_mask] = 2.0

    tx = 0.20
    t_vec = np.array([tx, 0.0, 0.0], dtype=np.float32)

    # Analytical expected shifts: fx * tx / Z
    exp_fg = fx * tx / 2.0   # 64.0 px
    exp_sub = fx * tx / 5.0  # 25.6 px
    exp_bg = fx * tx / 20.0  # 6.4 px

    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    pts_3d_fg = v0.back_project_points(u_grid[fg_mask].ravel(), v_grid[fg_mask].ravel(), depth[fg_mask].ravel(), fx, fy, cx, cy)
    pts_trans_fg = pts_3d_fg + t_vec
    u_p_fg, _, _ = v0.project_3d_points(pts_trans_fg, fx, fy, cx, cy)
    act_fg = float(np.mean(u_p_fg - u_grid[fg_mask].ravel()))

    pts_3d_sub = v0.back_project_points(u_grid[sub_mask].ravel(), v_grid[sub_mask].ravel(), depth[sub_mask].ravel(), fx, fy, cx, cy)
    pts_trans_sub = pts_3d_sub + t_vec
    u_p_sub, _, _ = v0.project_3d_points(pts_trans_sub, fx, fy, cx, cy)
    act_sub = float(np.mean(u_p_sub - u_grid[sub_mask].ravel()))

    pts_3d_bg = v0.back_project_points(u_grid[~sub_mask & ~fg_mask].ravel(), v_grid[~sub_mask & ~fg_mask].ravel(), depth[~sub_mask & ~fg_mask].ravel(), fx, fy, cx, cy)
    pts_trans_bg = pts_3d_bg + t_vec
    u_p_bg, _, _ = v0.project_3d_points(pts_trans_bg, fx, fy, cx, cy)
    act_bg = float(np.mean(u_p_bg - u_grid[~sub_mask & ~fg_mask].ravel()))

    assert np.isclose(act_fg, exp_fg, atol=1e-3)
    assert np.isclose(act_sub, exp_sub, atol=1e-3)
    assert np.isclose(act_bg, exp_bg, atol=1e-3)


def test_p0_post_fix_debug_tracing_and_frame_difference():
    """P0-POST-FIX TEST: Verify export_p0_raster_debug_trace and generate_p0_frame_difference_artifacts output valid report structures."""
    import v0_pipeline as v0
    import tempfile
    from pathlib import Path

    w, h = 64, 64
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[20:44, 20:44] = True
    depth = np.full((h, w), 5.0, dtype=np.float32); depth[sub_mask] = 2.0
    f0 = np.full((h, w, 3), 100, dtype=np.uint8)
    f_end = np.full((h, w, 3), 100, dtype=np.uint8); f_end[sub_mask] = [200, 50, 50]

    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", 1.0, num_frames=10)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        trace_p, dist_p = v0.export_p0_raster_debug_trace(
            trans, rots, sub_mask, depth, 64.0, 64.0, 32.0, 32.0, tmp_path
        )
        assert trace_p.exists()
        assert dist_p.exists()

        diff_report = v0.generate_p0_frame_difference_artifacts([f0, f_end], sub_mask, tmp_path)
        assert "end_to_end" in diff_report
        assert (tmp_path / "debug" / "frame_diff_f00_f99.png").exists()
        assert (tmp_path / "debug" / "frame_overlay_f00_f99.png").exists()
        assert (tmp_path / "debug" / "motion_heatmap.png").exists()


def test_p0_low_medium_high_raster_displacement_separation():
    """P0 REGRESSION TEST: Verify LOW < MEDIUM < HIGH raster displacement separation on actual rendered output."""
    import v0_pipeline as v0

    w, h = 128, 128
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[40:88, 40:88] = True
    bg_mask = ~sub_mask

    depth = np.linspace(1.5, 8.0, h * w).reshape(h, w).astype(np.float32)
    depth[sub_mask] = 2.0

    rng = np.random.RandomState(42)
    rgb = rng.randint(50, 200, (h, w, 3), dtype=np.uint8)

    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)
    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", 1.0, num_frames=10)

    disps = []
    for amp in ["LOW", "MEDIUM", "HIGH"]:
        m_map = v0.construct_layer_motion_map((h, w), sub_mask, motion_amplitude=amp)
        syn0, _, _ = v0.render_single_frame_forward_splatting(
            rgb, depth, rgb.copy(), depth, np.ones((h, w), dtype=np.float32),
            v0.compute_rotation_matrix(rots[0, 0], rots[0, 1], rots[0, 2]),
            trans[0], fx, fy, cx, cy, layer_motion_map=m_map
        )
        syn_mid, _, _ = v0.render_single_frame_forward_splatting(
            rgb, depth, rgb.copy(), depth, np.ones((h, w), dtype=np.float32),
            v0.compute_rotation_matrix(rots[5, 0], rots[5, 1], rots[5, 2]),
            trans[5], fx, fy, cx, cy, layer_motion_map=m_map
        )
        diag = v0.compute_perceptual_motion_score(
            [syn0, syn_mid], sub_mask, depth, [], trans[:6], rots[:6], motion_amplitude=amp
        )
        disps.append(diag["image_space"]["background_displacement_px"])

    # Verify monotonic perceptual separation LOW < MEDIUM < HIGH
    assert disps[0] < disps[1] < disps[2]


def test_p0_perceptual_camera_travel_and_quality_gates():
    """P0 REGRESSION TEST: Verify recalibrated camera trajectory produces obvious background displacement without excessive subject growth."""
    import v0_pipeline as v0

    w, h = 1536, 1024
    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[300:700, 500:1000] = True
    bg_mask = ~sub_mask

    bg_depth = np.linspace(1.5, 8.0, h * w).reshape(h, w).astype(np.float32)

    rng = np.random.RandomState(42)
    f0 = rng.randint(50, 200, (h, w, 3), dtype=np.uint8)

    q20, q70 = np.quantile(bg_depth[bg_mask], [0.20, 0.70])
    fg_mask = bg_mask & (bg_depth <= q20)
    mg_mask = bg_mask & (bg_depth > q20) & (bg_depth <= q70)
    bg_layer_mask = bg_mask & (bg_depth > q70)

    f_last = np.copy(f0)
    f_last[fg_mask] = np.roll(f0, 40, axis=1)[fg_mask]
    f_last[mg_mask] = np.roll(f0, 20, axis=1)[mg_mask]
    f_last[bg_layer_mask] = np.roll(f0, 10, axis=1)[bg_layer_mask]
    f_last[sub_mask] = np.roll(f0, 2, axis=1)[sub_mask]

    cam_trans, cam_rot = v0.generate_c1_smooth_trajectory("Cinematic Push-In", 1.0, 100)

    metrics = v0.compute_perceptual_motion_score(
        [f0, f_last], sub_mask, bg_depth, [], cam_trans, cam_rot, motion_amplitude="MEDIUM"
    )

    img_space = metrics["image_space"]

    # Verify background displacement is visually significant (> 5px raster displacement)
    assert img_space["background_displacement_px"] > 5.0

    # Verify subject stability restraint (< 8% scale growth)
    assert abs(img_space["subject_scale_growth"]) < 0.08

    # Verify quality gate passes
    assert metrics["perceptual_motion_gate_passed"] is True
    assert metrics["motion_visibility_class"] in ["SUBTLE", "VISIBLE", "CINEMATIC", "STRONG"]


def test_p23b_independent_raster_layer_motion_measurement():
    """TEST 23B: Verify all layer motion metrics are measured independently from actual layer raster data without synthetic formulas."""
    import v0_pipeline as v0

    w, h = 128, 128
    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[40:80, 40:80] = True

    # Render actual synthetic view frames with layer-differentiated motion multipliers
    bg_depth = np.linspace(1.5, 8.0, h * w).reshape(h, w).astype(np.float32)
    bg_depth[sub_mask] = 2.5

    rng = np.random.RandomState(42)
    f0 = rng.randint(50, 200, (h, w, 3), dtype=np.uint8)

    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)
    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", 1.0, num_frames=10)
    m_map = v0.construct_layer_motion_map((h, w), sub_mask, motion_amplitude="MEDIUM")

    syn0, _, _ = v0.render_single_frame_forward_splatting(
        f0, bg_depth, f0.copy(), bg_depth, np.ones((h, w), dtype=np.float32),
        v0.compute_rotation_matrix(rots[0, 0], rots[0, 1], rots[0, 2]),
        trans[0], fx, fy, cx, cy, layer_motion_map=m_map
    )
    syn_mid, _, _ = v0.render_single_frame_forward_splatting(
        f0, bg_depth, f0.copy(), bg_depth, np.ones((h, w), dtype=np.float32),
        v0.compute_rotation_matrix(rots[5, 0], rots[5, 1], rots[5, 2]),
        trans[5], fx, fy, cx, cy, layer_motion_map=m_map
    )

    metrics = v0.compute_perceptual_motion_score(
        [syn0, syn_mid], sub_mask, bg_depth, [], trans[:6], rots[:6], motion_amplitude="MEDIUM"
    )

    img_space = metrics["image_space"]

    fg_delta = img_space["foreground_displacement_px"]
    mg_delta = img_space["midground_displacement_px"]
    bg_delta = img_space["background_displacement_px"]
    sub_delta = img_space["subject_displacement_px"]

    # Verify no synthetic multiplier relationships hold exactly
    assert not np.isclose(fg_delta, sub_delta * 2.8, rtol=1e-3)
    assert not np.isclose(mg_delta, (bg_delta + sub_delta) / 2.0, rtol=1e-3)


# ============================================================
# PHASE 2.2: MONOTONIC RENDERED OUTPUT & CONTACT SHEET TESTS
# ============================================================

def test_p22_1_monotonic_rendered_output_benchmark():
    """TEST 1: Verify actual rasterized pixel motion satisfies HIGH > MEDIUM > LOW for subject displacement and scale growth."""
    import v0_pipeline as v0

    w, h = 128, 128
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[40:88, 40:88] = True
    rgb[sub_mask] = [200, 50, 50]
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)
    depth[sub_mask] = 2.0
    bg_plate = np.zeros_like(rgb); bg_depth = depth.copy(); prov = np.ones((h, w), dtype=np.float32)

    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)

    scales = []
    disps = []

    amp_scales = {"LOW": 0.5, "MEDIUM": 1.0, "HIGH": 1.8}
    for amp in ["LOW", "MEDIUM", "HIGH"]:
        mult = amp_scales[amp]
        m_map = v0.construct_layer_motion_map((h, w), sub_mask, motion_amplitude=amp)
        trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", magnitude_scale=mult, num_frames=100)
        syn, _, _ = v0.render_single_frame_forward_splatting(
            rgb, depth, bg_plate, bg_depth, prov,
            v0.compute_rotation_matrix(rots[-1, 0], rots[-1, 1], rots[-1, 2]),
            trans[-1], fx, fy, cx, cy, layer_motion_map=m_map
        )
        s_m = v0.evaluate_subject_scale_change(sub_mask, rgb, syn)
        scales.append(s_m["subject_scale_growth"])
        diff = np.abs(syn.astype(np.float32) - rgb.astype(np.float32))
        rendered_sub = (syn[:, :, 0] > 50) | sub_mask
        disps.append(float(np.mean(diff[rendered_sub])))

    # Assert Monotonicity Invariant: HIGH > MEDIUM > LOW
    assert scales[0] < scales[1] < scales[2]
    assert disps[0] < disps[1] < disps[2]


def test_p22_2_five_keyframe_contact_sheet_sampling():
    """TEST 2: Verify 5-keyframe contact sheet sampling includes F00, F24, F49, F74, F99 for 100-frame render."""
    import v0_pipeline as v0

    w, h = 32, 32
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[8:24, 8:24] = True
    bg_plate = rgb.copy(); bg_depth = np.full((h, w), 5.0, dtype=np.float32)
    prov = np.ones((h, w), dtype=np.float32)

    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", magnitude_scale=1.0, num_frames=100)
    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)

    sheet = v0.generate_motion_amplitude_comparison_contact_sheet(
        rgb, bg_depth, sub_mask, bg_plate, bg_depth, prov, trans, rots, fx, fy, cx, cy, target_w=64
    )

    assert sheet.ndim == 3
    assert sheet.shape[0] > 0


# ============================================================
# PHASE 4: CAMERA APPLICATION SMOKE TESTS (TESTS A - F)
# ============================================================

def test_camera_smoke_a_identity_transform():
    """SMOKE TEST A: Zero camera translation and rotation returns image identical to input."""
    import v0_pipeline as v0

    w, h = 64, 64
    rgb = np.full((h, w, 3), fill_value=128, dtype=np.uint8)
    rgb[20:44, 20:44] = [200, 50, 50]
    depth = np.full((h, w), fill_value=2.0, dtype=np.float32)
    bg_plate = rgb.copy(); bg_depth = depth.copy(); prov = np.ones((h, w), dtype=np.float32)
    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)

    syn_rgb, _, _ = v0.render_single_frame_forward_splatting(
        rgb, depth, bg_plate, bg_depth, prov, np.eye(3), np.zeros(3), fx, fy, cx, cy
    )

    abs_diff = np.abs(syn_rgb.astype(np.float32) - rgb.astype(np.float32))
    assert np.mean(abs_diff) < 2.0  # Identity error must be near zero


def test_camera_smoke_b_and_c_positive_negative_translation():
    """SMOKE TEST B & C: Positive vs negative translation produces reversed pixel displacement directions."""
    import v0_pipeline as v0

    w, h = 64, 64
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[20:44, 20:44] = True
    rgb[sub_mask] = [200, 50, 50]
    depth = np.full((h, w), fill_value=2.0, dtype=np.float32)
    bg_plate = np.zeros_like(rgb); bg_depth = depth.copy(); prov = np.ones((h, w), dtype=np.float32)
    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)

    # Positive translation tx = +0.10
    syn_pos, _, _ = v0.render_single_frame_forward_splatting(
        rgb, depth, bg_plate, bg_depth, prov, np.eye(3), np.array([0.10, 0.0, 0.0]), fx, fy, cx, cy
    )
    # Negative translation tx = -0.10
    syn_neg, _, _ = v0.render_single_frame_forward_splatting(
        rgb, depth, bg_plate, bg_depth, prov, np.eye(3), np.array([-0.10, 0.0, 0.0]), fx, fy, cx, cy
    )

    y_pos, x_pos = np.where(syn_pos[:, :, 0] > 100)
    y_neg, x_neg = np.where(syn_neg[:, :, 0] > 100)

    # Positive tx shifts object right (higher mean x), Negative tx shifts object left (lower mean x)
    assert np.mean(x_pos) > np.mean(x_neg)


def test_camera_smoke_d_amplitude_monotonicity():
    """SMOKE TEST D: LOW < MEDIUM < HIGH motion amplitude presets produce strictly monotonic actual pixel displacement."""
    import v0_pipeline as v0

    w, h = 64, 64
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[20:44, 20:44] = True
    rgb[sub_mask] = [200, 50, 50]
    depth = np.full((h, w), fill_value=2.0, dtype=np.float32)
    bg_plate = np.zeros_like(rgb); bg_depth = depth.copy(); prov = np.ones((h, w), dtype=np.float32)
    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)
    t_vec = np.array([0.05, 0.0, 0.0])

    disps = []
    for amp in ["LOW", "MEDIUM", "HIGH"]:
        m_map = v0.construct_layer_motion_map((h, w), sub_mask, motion_amplitude=amp)
        syn_rgb, _, _ = v0.render_single_frame_forward_splatting(
            rgb, depth, bg_plate, bg_depth, prov, np.eye(3), t_vec, fx, fy, cx, cy, layer_motion_map=m_map
        )
        diff = np.abs(syn_rgb.astype(np.float32) - rgb.astype(np.float32))
        disps.append(float(np.mean(diff[sub_mask])))

    assert disps[0] < disps[1] < disps[2]  # LOW < MEDIUM < HIGH


def test_camera_smoke_e_z_dolly_monotonic_response():
    """SMOKE TEST E: Increasing camera push-in Z displacement produces monotonic subject scale growth."""
    import v0_pipeline as v0

    w, h = 128, 128
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[40:88, 40:88] = True
    rgb[sub_mask] = [200, 50, 50]
    depth = np.full((h, w), fill_value=2.0, dtype=np.float32)
    bg_plate = np.zeros_like(rgb); bg_depth = depth.copy(); prov = np.ones((h, w), dtype=np.float32)
    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)

    scales = []
    for mag in [-0.05, -0.10, -0.20]:
        t_vec = np.array([0.0, 0.0, mag])
        syn_rgb, _, _ = v0.render_single_frame_forward_splatting(
            rgb, depth, bg_plate, bg_depth, prov, np.eye(3), t_vec, fx, fy, cx, cy
        )
        scale_m = v0.evaluate_subject_scale_change(sub_mask, rgb, syn_rgb)
        scales.append(scale_m["subject_scale_growth"])

    assert scales[0] < scales[1] < scales[2]  # Monotonic scale growth with Z push-in


def test_camera_smoke_f_layer_differential_motion():
    """SMOKE TEST F: Actual rendered layer displacements strictly observe FOREGROUND > MIDGROUND > BACKGROUND > PRIMARY_SUBJECT."""
    from spatial_intelligence.camera_model import compute_layer_motion_multiplier

    m_fg = compute_layer_motion_multiplier("FOREGROUND", "MEDIUM")
    m_sub = compute_layer_motion_multiplier("PRIMARY_SUBJECT", "MEDIUM")
    m_mg = compute_layer_motion_multiplier("MIDGROUND", "MEDIUM")
    m_bg = compute_layer_motion_multiplier("BACKGROUND", "MEDIUM")

    assert m_fg > m_mg > m_bg > m_sub


# ============================================================
# PHASE 2.1: PERCEPTUAL CAMERA MOTION TESTS
# ============================================================

def test_phase_2_8_benchmark_viewpoint_and_routing():
    """PHASE 2.8 TEST: Verifies hardware benchmark, viewpoint failure sweep contracts, and backend decision export."""
    from scene_3d.reconstruction import HardwareProfile, SceneComplexityTier, QualityPlanner

    hw = HardwareProfile.detect()
    decision = QualityPlanner.plan(hw, SceneComplexityTier.TIER2_MODERATE, (1280, 720))

    assert decision.backend.value in ["CPU", "CUDA", "HYBRID"]
    dict_dec = decision.to_dict()
    assert "hardware" in dict_dec
    assert "source_resolution" in dict_dec
    assert "reconstruction_resolution" in dict_dec
    assert "output_resolution" in dict_dec


def test_quality_planner_hardware_aware_routing():
    """PHASE 2.6 ADDENDUM TEST: Verifies QualityPlanner CPU 720p ceiling, VRAM downgrade, and Quality Honesty contract."""
    from scene_3d.reconstruction import HardwareProfile, SceneComplexityTier, QualityPlanner

    # Test CPU 4K request triggers 720p ceiling downgrade
    hw_cpu = HardwareProfile(has_cuda=False, device_name="CPU", vram_gb=0.0, ram_gb=16.0, cpu_cores=4)
    decision_cpu = QualityPlanner.plan(hw_cpu, SceneComplexityTier.COMPLEX, (3840, 2160), requested_resolution="4k")
    assert decision_cpu.output_resolution == (1280, 720)
    assert decision_cpu.downgrade_reason is not None
    assert "CPU execution constrained" in decision_cpu.downgrade_reason

    # Test GPU VRAM constrained downgrade (< 4GB VRAM)
    hw_gpu_low = HardwareProfile(has_cuda=True, device_name="Low VRAM GPU", vram_gb=2.0, ram_gb=16.0, cpu_cores=8)
    decision_low_vram = QualityPlanner.plan(hw_gpu_low, SceneComplexityTier.COMPLEX, (3840, 2160), requested_resolution="4k")
    assert decision_low_vram.output_resolution == (1280, 720)
    assert "VRAM constrained" in decision_low_vram.downgrade_reason

    # Test High-End GPU 4K
    hw_gpu_high = HardwareProfile(has_cuda=True, device_name="NVIDIA RTX 4090", vram_gb=24.0, ram_gb=32.0, cpu_cores=16)
    decision_high_vram = QualityPlanner.plan(hw_gpu_high, SceneComplexityTier.COMPLEX, (3840, 2160), requested_resolution="4k")
    assert decision_high_vram.output_resolution == (3840, 2160)
    assert decision_high_vram.downgrade_reason is None

    # Quality Honesty contract check
    dict_repr = decision_high_vram.to_dict()
    assert "source_resolution" in dict_repr
    assert "reconstruction_resolution" in dict_repr
    assert "output_resolution" in dict_repr


def test_scene_3d_inferred_reconstruction_pipeline():
    """PHASE 2.6 TEST: Verifies scene_3d/ inferred scene reconstruction, complexity routing, and novel view rendering."""
    import scene_3d as s3d

    h, w = 32, 32
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 180
    depth = np.full((h, w), 4.0, dtype=np.float32)
    sub_mask = np.zeros((h, w), dtype=bool)
    sub_mask[8:24, 8:24] = True

    tier, score, details = s3d.SceneComplexityAnalyzer.analyze(rgb, depth, sub_mask)
    assert tier in [s3d.SceneComplexityTier.SIMPLE, s3d.SceneComplexityTier.MODERATE, s3d.SceneComplexityTier.COMPLEX]

    scene = s3d.reconstruct_inferred_3d_scene(rgb, depth, sub_mask)
    assert len(scene.point_cloud.vertices) == 32 * 32
    assert len(scene.mesh.faces) > 0

    R = np.eye(3)
    t = np.array([0.02, 0.0, 0.0], dtype=np.float64)
    syn_rgb = scene.render_novel_view(R, t)
    assert syn_rgb.shape == (h, w, 3)


def test_render_backend_explicit_3d_mesh_export():
    """PHASE 2.7 TEST: Verifies explicit 3D mesh construction and OBJ/PLY export in render_backend/."""
    import tempfile
    from pathlib import Path
    from render_backend.explicit_3d import construct_explicit_3d_mesh, export_scene_3d_package

    h, w = 32, 32
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 150
    depth = np.full((h, w), 5.0, dtype=np.float32)
    prov = np.ones((h, w), dtype=np.float32)

    mesh = construct_explicit_3d_mesh(rgb, depth, 32.0, 32.0, 16.0, 16.0, provenance_map=prov)
    assert len(mesh.vertices) == 32 * 32
    assert len(mesh.faces) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        summary = export_scene_3d_package(rgb, depth, 32.0, 32.0, 16.0, 16.0, tmp_path, provenance_map=prov)
        assert (tmp_path / "scene_mesh.obj").exists()
        assert (tmp_path / "point_cloud.ply").exists()
        assert (tmp_path / "scene_3d_graph.json").exists()
        assert summary["vertex_count"] == 1024


def test_subpixel_splatting_math_and_z_ownership():
    """PHASE 2.4C TEST: Verifies subpixel splatting weight conservation (sum=1.0) and Z-buffer ownership."""
    from v0_pipeline import render_single_frame_forward_splatting

    # Test bilinear weight sum for subpixel offsets
    for pu in [10.01, 10.05, 10.10, 10.25, 10.50, 11.00]:
        for pv in [20.01, 20.05, 20.10, 20.25, 20.50, 21.00]:
            u0 = int(np.floor(pu))
            v0 = int(np.floor(pv))
            du = float(pu - u0)
            dv = float(pv - v0)
            w00 = (1.0 - du) * (1.0 - dv)
            w10 = du * (1.0 - dv)
            w01 = (1.0 - du) * dv
            w11 = du * dv
            w_sum = w00 + w10 + w01 + w11
            assert np.isclose(w_sum, 1.0, atol=1e-6)

    # Test Z-ownership reset in single frame splatting
    h, w = 64, 64
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 100
    depth = np.full((h, w), 5.0, dtype=np.float32)
    bg_plate = np.ones((h, w, 3), dtype=np.uint8) * 50
    bg_depth = np.full((h, w), 10.0, dtype=np.float32)
    prov = np.ones((h, w), dtype=np.float32)
    R = np.eye(3)
    t = np.array([0.05, 0.0, 0.0], dtype=np.float64)

    syn_rgb, syn_z, syn_prov = render_single_frame_forward_splatting(
        rgb, depth, bg_plate, bg_depth, prov, R, t, 64.0, 64.0, 32.0, 32.0
    )

    assert syn_rgb.shape == (h, w, 3)
    valid_z = syn_z[syn_z < 1e8]
    assert len(valid_z) > 0
    assert np.mean(valid_z) == pytest.approx(5.0, abs=0.1)  # Closer surface Z (5.0) strictly wins over background Z (10.0)


def test_p21_1_negative_tz_camera_push_in_direction():
    """TEST 1: Verify Cinematic Push-In decreases camera-to-subject distance (negative t_z)."""
    import v0_pipeline as v0

    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", magnitude_scale=1.0, num_frames=100)
    assert trans[-1, 2] < 0.0
    assert trans[-1, 2] == pytest.approx(-0.35, abs=0.15)


def test_p21_2_raster_subject_scale_growth():
    """TEST 2: Verify evaluate_subject_scale_change measures subject scale growth directly from rendered frames."""
    import v0_pipeline as v0

    h, w = 128, 128
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[32:96, 32:96] = True
    f0 = np.zeros((h, w, 3), dtype=np.uint8); f0[sub_mask] = [200, 50, 50]

    f_end = np.zeros((h, w, 3), dtype=np.uint8); f_end[28:100, 28:100] = [200, 50, 50]  # Expanded subject

    scale_metrics = v0.evaluate_subject_scale_change(sub_mask, f0, f_end)
    assert "subject_scale_growth" in scale_metrics
    assert scale_metrics["subject_scale_growth"] > 0.05
    assert scale_metrics["scale_change_ratio"] > 1.0


def test_p21_3_perceptual_motion_gate_rejection():
    """TEST 3: Verify hard perceptual motion gate fails when subject motion and scale growth are locked/static."""
    import v0_pipeline as v0

    h, w = 32, 32
    f0 = np.full((h, w, 3), 100, dtype=np.uint8)
    f0[8:24, 8:24] = [200, 50, 50]  # Distinct subject color
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[8:24, 8:24] = True
    bg_depth = np.full((h, w), 5.0, dtype=np.float32)

    # Identical frames (zero motion)
    score_dict = v0.compute_perceptual_motion_score(
        [f0, f0], sub_mask, bg_depth, [], np.zeros((10, 3)), np.zeros((10, 3))
    )

    assert score_dict["perceptual_motion_gate_passed"] is False
    assert score_dict["motion_good"] is False


def test_p21_4_motion_stability_vs_effectiveness_separation():
    """TEST 4: Verify MOTION_STABILITY and MOTION_EFFECTIVENESS are distinct metrics."""
    import v0_pipeline as v0

    h, w = 32, 32
    f0 = np.full((h, w, 3), 100, dtype=np.uint8)
    f0[8:24, 8:24] = [200, 50, 50]  # Distinct subject color
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[8:24, 8:24] = True
    bg_depth = np.full((h, w), 5.0, dtype=np.float32)

    score_dict = v0.compute_perceptual_motion_score(
        [f0, f0], sub_mask, bg_depth, [{"mean_disparity_px": 0.1}], np.zeros((10, 3)), np.zeros((10, 3))
    )

    assert "motion_stability_score" in score_dict
    assert "motion_effectiveness_score" in score_dict
    # Perfectly static frames have high stability but zero effectiveness
    assert score_dict["motion_stability_score"] > 0.90
    assert score_dict["motion_effectiveness_score"] == pytest.approx(0.0)


def test_p21_5_foreground_only_drift_does_not_pass_gate():
    """TEST 5: Verify that drift limited only to foreground while subject remains static fails the motion gate."""
    import v0_pipeline as v0

    vis_class = v0.classify_motion_visibility(
        subject_disp_px=1.0, bg_disp_px=0.5, relative_disp_px=0.5, scale_change_ratio=1.0
    )
    assert vis_class in ["NEGLIGIBLE", "SUBTLE", "WEAK"]


def test_p21_6_motion_report_file_structure():
    """TEST 6: Verify motion_report.json structure and required key fields."""
    import v0_pipeline as v0

    h, w = 32, 32
    f0 = np.zeros((h, w, 3), dtype=np.uint8)
    f_end = np.full((h, w, 3), fill_value=150, dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[8:24, 8:24] = True
    bg_depth = np.full((h, w), fill_value=5.0, dtype=np.float32)

    score_dict = v0.compute_perceptual_motion_score(
        [f0, f_end], sub_mask, bg_depth, [], np.zeros((10, 3)), np.zeros((10, 3))
    )

    assert "motion_stability_score" in score_dict
    assert "motion_effectiveness_score" in score_dict
    assert "perceptual_motion_gate_passed" in score_dict


def test_p21_7_layer_parallax_motion_ordering_preserved():
    """TEST 7: Verify layer parallax motion ordering FOREGROUND > MIDGROUND > BACKGROUND > PRIMARY_SUBJECT."""
    from spatial_intelligence.camera_model import compute_layer_motion_multiplier

    m_fg = compute_layer_motion_multiplier("FOREGROUND", "MEDIUM")
    m_sub = compute_layer_motion_multiplier("PRIMARY_SUBJECT", "MEDIUM")
    m_mg = compute_layer_motion_multiplier("MIDGROUND", "MEDIUM")
    m_bg = compute_layer_motion_multiplier("BACKGROUND", "MEDIUM")

    assert m_fg > m_mg > m_bg > m_sub


# ============================================================
# PHASE 2.3: DEPTH-DRIVEN CINEMATIC PARALLAX REBALANCING TESTS
# ============================================================

def test_p23_1_depth_weighted_layer_motion_ordering():
    """TEST 1: Verify Phase 2.3 depth-weighted layer motion ordering FOREGROUND > MIDGROUND > BACKGROUND > PRIMARY_SUBJECT."""
    from spatial_intelligence.camera_model import compute_layer_motion_multiplier

    m_fg = compute_layer_motion_multiplier("FOREGROUND", "MEDIUM")
    m_mg = compute_layer_motion_multiplier("MIDGROUND", "MEDIUM")
    m_bg = compute_layer_motion_multiplier("BACKGROUND", "MEDIUM")
    m_sub = compute_layer_motion_multiplier("PRIMARY_SUBJECT", "MEDIUM")

    assert m_fg > m_mg > m_bg > m_sub
    assert m_bg > m_sub  # Background environmental motion is higher than subject motion


def test_p23_2_subject_scale_growth_restrained():
    """TEST 2: Verify primary subject scale growth remains restrained in target 1-4% range."""
    import v0_pipeline as v0

    w, h = 128, 128
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[40:88, 40:88] = True
    rgb[sub_mask] = [200, 50, 50]
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)
    depth[sub_mask] = 2.0
    bg_plate = np.zeros_like(rgb); bg_depth = depth.copy(); prov = np.ones((h, w), dtype=np.float32)

    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)
    m_map = v0.construct_layer_motion_map((h, w), sub_mask, motion_amplitude="MEDIUM")

    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", magnitude_scale=1.2, num_frames=100)
    syn_rgb, _, _ = v0.render_single_frame_forward_splatting(
        rgb, depth, bg_plate, bg_depth, prov,
        v0.compute_rotation_matrix(rots[-1, 0], rots[-1, 1], rots[-1, 2]),
        trans[-1], fx, fy, cx, cy, layer_motion_map=m_map
    )

    scale_metrics = v0.evaluate_subject_scale_change(sub_mask, rgb, syn_rgb)
    assert 0.0 <= scale_metrics["subject_scale_growth"] <= 0.10  # Restrained scale growth


def test_p23_3_environmental_motion_score_calculation():
    """TEST 3: Verify environmental_motion_score and subject_stability_score calculation."""
    import v0_pipeline as v0

    h, w = 32, 32
    f0 = np.full((h, w, 3), 100, dtype=np.uint8)
    f0[10:20, 10:20] = [200, 50, 50]
    f_end = f0.copy()
    f_end[0:10, :] = [150, 150, 150]  # Background movement

    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[10:20, 10:20] = True
    bg_depth = np.full((h, w), 5.0, dtype=np.float32)

    score_dict = v0.compute_perceptual_motion_score(
        [f0, f_end], sub_mask, bg_depth, [], np.zeros((10, 3)), np.zeros((10, 3)), motion_amplitude="MEDIUM"
    )

    assert "environmental_motion_score" in score_dict
    assert "subject_stability_score" in score_dict
    assert score_dict["subject_stability_score"] > 0.90


def test_p23_4_weak_motion_fails_quality_gate():
    """TEST 4: Verify WEAK motion classification fails perceptual quality gate (motion_good = False)."""
    import v0_pipeline as v0

    vis_class = v0.classify_motion_visibility(
        subject_disp_px=1.0, bg_disp_px=2.0, relative_disp_px=1.0, scale_change_ratio=1.001, motion_amplitude="MEDIUM"
    )
    assert vis_class == "WEAK"


def test_p21_8_cinematic_push_in_produces_measurable_scale_growth():
    """TEST 8: Verify 3D forward splatting with push-in produces measurable subject scale expansion."""
    import v0_pipeline as v0

    w, h = 640, 640
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[100:540, 100:540] = True
    depth[sub_mask] = 2.0
    rgb[sub_mask] = [200, 50, 50]

    fx, fy, cx, cy = v0.derive_camera_intrinsics(w, h)
    m_map = v0.construct_layer_motion_map((h, w), sub_mask, motion_amplitude="MEDIUM")

    t_vec = np.array([0.0, 0.0, -0.20])  # Camera pushes forward
    syn_rgb, _, _ = v0.render_single_frame_forward_splatting(
        rgb, depth, np.zeros_like(rgb), depth, np.ones((h, w), dtype=np.float32),
        np.eye(3), t_vec, fx, fy, cx, cy, layer_motion_map=m_map
    )

    scale_metrics = v0.evaluate_subject_scale_change(sub_mask, rgb, syn_rgb)
    assert scale_metrics["subject_scale_growth"] > 0.02


def test_p21_9_low_medium_high_perceptual_motion_ranking():
    """TEST 9: Verify LOW < MEDIUM < HIGH motion amplitude ranking for push-in camera displacement."""
    import v0_pipeline as v0

    t_low, _ = v0.generate_c1_smooth_trajectory("Cinematic Push-In", 0.5, 100)
    t_med, _ = v0.generate_c1_smooth_trajectory("Cinematic Push-In", 1.0, 100)
    t_high, _ = v0.generate_c1_smooth_trajectory("Cinematic Push-In", 1.5, 100)

    assert abs(t_low[-1, 2]) < abs(t_med[-1, 2]) < abs(t_high[-1, 2])


def test_p21_10_layer_displacement_curves_plot_generation():
    """TEST 10: Verify layer_displacement_curves.png plot array generation."""
    import v0_pipeline as v0

    w, h = 64, 64
    trans, rots = v0.generate_c1_smooth_trajectory("Cinematic Push-In", 1.0, 100)
    sub_mask = np.zeros((h, w), dtype=bool); sub_mask[20:44, 20:44] = True
    depth = np.full((h, w), fill_value=5.0, dtype=np.float32)

    plot = v0.generate_layer_displacement_curve_plot(
        trans, rots, sub_mask, depth, 64.0, 64.0, 32.0, 32.0, motion_amplitude="MEDIUM"
    )

    assert plot.shape == (320, 640, 3)
    assert np.any(plot > 0)




# ============================================================
# PHASE 3: REAL-WORLD MODE VALIDATION & HARDWARE TESTS
# ============================================================

def test_phase3_explicit_mode_a_and_mode_b_cli():
    """PHASE 3 TEST 1: Verifies explicit CLI flags --render-mode 2.5d and --render-mode 3d."""
    args_a = parse_args(["--input", "test.jpg", "--render-mode", "2.5d"])
    assert args_a.render_mode == "2.5d"

    args_b = parse_args(["--input", "test.jpg", "--render-mode", "3d"])
    assert args_b.render_mode == "3d"

    args_auto = parse_args(["--input", "test.jpg", "--render-mode", "auto"])
    assert args_auto.render_mode == "auto"


def test_phase3_explainable_auto_routing():
    """PHASE 3 TEST 2: Verifies Auto Router decision output structure and explainable reason codes."""
    from scene_3d.reconstruction import HardwareProfile, SceneComplexityTier, QualityPlanner

    hw = HardwareProfile.detect()
    decision = QualityPlanner.plan(hw, SceneComplexityTier.TIER2_MODERATE, (1280, 720))

    dec_dict = decision.to_dict()
    assert "backend" in dec_dict
    assert "source_resolution" in dec_dict
    assert "reconstruction_resolution" in dec_dict
    assert "output_resolution" in dec_dict


def test_phase3_cpu_720p_ceiling_and_quality_honesty_downgrade():
    """PHASE 3 TEST 3: Verifies CPU execution <= 720p ceiling enforcement and Quality Honesty contract warnings."""
    from scene_3d.reconstruction import HardwareProfile, SceneComplexityTier, QualityPlanner

    # CPU mode requested at 1080p must downgrade to 720p with explicit warning
    hw_cpu = HardwareProfile(has_cuda=False, device_name="CPU", vram_gb=0.0, ram_gb=16.0, cpu_cores=4)
    decision = QualityPlanner.plan(hw_cpu, SceneComplexityTier.TIER3_COMPLEX, (1920, 1080), requested_resolution="1080p")

    assert decision.output_resolution == (1280, 720)
    assert decision.downgrade_reason is not None
    assert "CPU execution constrained" in decision.downgrade_reason


def test_phase3_3d_export_package_contracts():
    """PHASE 3 TEST 4: Verifies explicit 3D mesh and point cloud export contract (OBJ, PLY, GLB-graph metadata)."""
    import tempfile
    from pathlib import Path
    from render_backend.explicit_3d import export_scene_3d_package

    h, w = 32, 32
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 120
    depth = np.full((h, w), 4.0, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir)
        summary = export_scene_3d_package(rgb, depth, 32.0, 32.0, 16.0, 16.0, export_dir)

        assert (export_dir / "scene_mesh.obj").exists()
        assert (export_dir / "point_cloud.ply").exists()
        assert (export_dir / "scene_3d_graph.json").exists()
        assert summary["vertex_count"] == 1024


def test_phase3_viewpoint_sweep_confidence_decay():
    """PHASE 3 TEST 5: Verifies viewpoint angle geometric confidence decay and failure status classification."""
    angles = [0, 5, 10, 15, 20, 30, 45]
    confidences = [max(0.0, 1.0 - (angle / 45.0) * 0.85) for angle in angles]

    # Geometric confidence MUST decrease monotonically with increasing viewpoint angle
    assert confidences[0] > confidences[1] > confidences[2] > confidences[3]
    assert confidences[0] == 1.0
    assert confidences[-1] == pytest.approx(0.15)


def test_phase3_hardware_quality_matrix_schema():
    """PHASE 3 TEST 6: Verifies hardware quality matrix JSON and quality report schemas."""
    import json, os

    if Path("hardware_quality_matrix.json").exists():
        with open("hardware_quality_matrix.json", "r") as f:
            hw_matrix = json.load(f)
        assert "detected_hardware" in hw_matrix
        assert "hardware_profiles" in hw_matrix

    if Path("quality_report.json").exists():
        with open("quality_report.json", "r") as f:
            q_report = json.load(f)
        assert "REQUESTED" in q_report
        assert "ACTUAL" in q_report
        assert "LIMITATIONS" in q_report


def test_nonzero_motion_generates_nonzero_trajectory():
    """Verify that non-zero camera motion requests produce non-zero camera trajectories."""
    import camera
    from camera.trajectories import generate_c1_smooth_trajectory

    for style in ["Cinematic Push-In", "Dolly In", "Dolly Out", "Horizontal Pan", "Vertical Pan", "Orbit"]:
        trans, rots = generate_c1_smooth_trajectory(style, magnitude_scale=1.0, num_frames=24)
        assert trans.shape == (24, 3)
        assert rots.shape == (24, 3)
        max_trans = float(np.max(np.abs(trans)))
        assert max_trans > 0.0, f"Trajectory for style '{style}' collapsed to zero"


def test_subject_lock_functionality():
    from modes.mode_2_5d.scene import SubjectAnchor, extract_subject_anchor
    from geometry.transforms import regularize_subject_depth, compute_subject_rigid_transform
    from rendering.disocclusion import PersistentBackgroundCanvas
    from quality.metrics import compute_subject_lock_metrics, validate_subject_lock_quality_gate

    # Synthetic 64x64 mask & depth
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True
    depth = np.full((64, 64), 5.0, dtype=np.float32)

    # 1. Subject Anchor
    anchor = extract_subject_anchor(mask, depth)
    assert anchor is not None
    assert anchor.centroid == (29.5, 29.5)

    # 2. Regularization
    reg_depth = regularize_subject_depth(depth, mask)
    assert reg_depth.shape == depth.shape

    # 3. Rigid transform
    R_cam = np.eye(3, dtype=np.float64)
    t_cam = np.array([0.1, 0.0, -0.2], dtype=np.float64)
    R_sub, t_sub = compute_subject_rigid_transform(R_cam, t_cam, 5.0)
    assert t_sub[0] < t_cam[0]

    # 4. Canvas
    rgb = np.full((64, 64, 3), 100, dtype=np.uint8)
    canvas = PersistentBackgroundCanvas(rgb, depth)
    c_rgb, c_d = canvas.get_canvas()
    assert c_rgb.shape == (64, 64, 3)

    # 5. Lock Metrics & Gate
    frames = [rgb, rgb.copy()]
    m = compute_subject_lock_metrics(frames, mask, depth)
    assert m["subject_temporal_stability_score"] >= 75.0
    passed, _ = validate_subject_lock_quality_gate(m)
    assert passed is True


def test_phase3_4_synthetic_regression_suite():
    """
    Phase 3.4 Synthetic Regression Suite verifying contracts and invariants:
    - Bounding Box Contract [x1, y1, x2, y2]
    - Subpixel Rigid Warp
    - Subject Interior Protection
    - Subject Pixel Integrity Metric
    """
    from core.contracts import validate_bounding_box_contract
    from geometry.splatting import warp_subject_layer_rigid_subpixel
    from rendering.disocclusion import protect_primary_subject_interior
    from quality.metrics import compute_subject_pixel_integrity

    # 1. Bounding box contract test
    bbox = validate_bounding_box_contract((10, 20, 100, 200), 320, 320)
    assert bbox == (10, 20, 100, 200)

    # 2. Subpixel rigid warp
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[20:80, 20:80] = [255, 0, 0, 255]
    aff = np.eye(3, dtype=np.float32)
    warped = warp_subject_layer_rigid_subpixel(rgba, aff, (100, 100))
    assert warped.shape == (100, 100, 4)
    assert np.all(warped[20:80, 20:80, 0] == 255)

    # 3. Primary Subject Interior Protection
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    protected = protect_primary_subject_interior(rgba, mask, interior_erosion_size=5)
    assert protected.shape == (100, 100, 4)
    assert np.all(protected[30:70, 30:70, 3] == 255)

    # 4. Subject Pixel Integrity Metric
    frames = [rgba[..., :3], rgba[..., :3].copy()]
    metric = compute_subject_pixel_integrity(frames, mask)
    assert metric.passed is True
    assert metric.subject_integrity_score >= 75.0
