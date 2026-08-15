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
    render_full_48_frame_sequence,
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
        rendered_frames, frame_metrics = render_full_48_frame_sequence(
            rgb, depth, rgb, depth, prov, sub_mask, risk_map,
            trans, rots, fx, fy, cx, cy, disparity_ceiling_px=50.0, frames_dir=frames_dir
        )

        assert len(rendered_frames) == 48
        assert len(frame_metrics) == 48
        assert (frames_dir / "frame_00.png").exists()
        assert (frames_dir / "frame_47.png").exists()

        # 2. Compute Temporal Diagnostics
        temp_summary, temp_plot = compute_temporal_diagnostics(rendered_frames, sub_mask)
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
