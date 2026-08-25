"""
v0_pipeline.py - Backwards-Compatible Thin CLI Entry Point.
"""

from inference import (
    get_device, DEPTH_MODEL_ID, SAM2_MODEL_ID, SAM2_CKPT_FILENAME, SAM2_CONFIG_NAME,
    load_depth_anything_v2, load_sam2, infer_raw_depth, handle_depth_outliers_and_normalize,
    edge_aware_depth_refinement, compute_depth_confidence_map, compute_mask_iou,
    generate_sam2_candidate_masks, compute_candidate_features, score_and_rank_candidates,
    check_and_group_candidates, evaluate_subject_selection_confidence_gate,
    generate_candidate_masks_contact_sheet, export_candidate_selection_json,
    segment_subject_sam2, validate_subject_mask, refine_and_dilate_subject_mask,
    apply_depth_aware_edge_feathering, compute_boundary_risk_map,
)
from rendering import (
    reconstruct_background_rgb, complete_background_depth, compute_provenance_map,
    verify_zero_motion_identity, run_micro_motion_sweep, synthesize_micro_motion_frame,
    render_full_frame_sequence, verify_ffmpeg, extract_and_verify_mp4_frames, encode_and_verify_mp4,
    render_phase_e_representative_keyframes,
)
from quality import (
    compute_subject_rigidity_metrics, classify_motion_visibility, evaluate_subject_scale_change,
    compute_perceptual_motion_score, analyze_image_space_motion_and_subject_fidelity,
    compute_temporal_diagnostics, export_temporal_motion_profile, export_p0_raster_debug_trace,
    generate_projected_vs_raster_trajectory_plot, generate_pixel_trajectory_plot,
    generate_camera_vs_raster_motion_plot, generate_camera_path_plot, generate_layer_displacement_curve_plot,
    generate_discontinuity_rejection_map, analyze_zero_motion_errors, generate_micro_sweep_contact_sheet,
    generate_subject_coherence_diagnostics, generate_phase_d_crop_diagnostics, generate_visual_review_contact_sheet,
    generate_motion_amplitude_comparison_contact_sheet, generate_p0_frame_difference_artifacts,
    generate_phase_1_7_multi_row_contact_sheet, generate_visual_review_diagnostics_sheet,
    generate_final_contact_sheet, generate_phase_e_keyframe_contact_sheet,
)
from camera import (
    derive_camera_intrinsics, generate_c1_smooth_trajectory, compute_safety_margins,
    run_trajectory_magnitude_sweep, plan_safe_motion_trajectory,
)
from geometry import (
    back_project_points, project_3d_points, compute_rotation_matrix, transform_3d_points,
    construct_layer_motion_map, render_single_frame_forward_splatting, deterministic_z_buffer_update,
)
from output import (
    compute_image_sha256, validate_and_load_image, setup_cache_directory, setup_output_directories,
    save_phase_b_diagnostic_artifacts, save_phase_c_diagnostic_artifacts,
    save_phase_d_validation_artifacts, save_phase_d_diagnostic_artifacts, save_phase_e_artifacts,
)
from app.cli import parse_args, main

if __name__ == "__main__":
    main()
