"""
Quality Package Entry Point.
"""

from .metrics import (
    compute_subject_rigidity_metrics,
    classify_motion_visibility,
    evaluate_subject_scale_change,
    compute_perceptual_motion_score,
    analyze_image_space_motion_and_subject_fidelity,
    compute_temporal_diagnostics,
)
from .plots import (
    export_temporal_motion_profile,
    export_p0_raster_debug_trace,
    generate_projected_vs_raster_trajectory_plot,
    generate_pixel_trajectory_plot,
    generate_camera_vs_raster_motion_plot,
    generate_camera_path_plot,
    generate_layer_displacement_curve_plot,
)
from .diagnostics import (
    generate_discontinuity_rejection_map,
    analyze_zero_motion_errors,
    generate_micro_sweep_contact_sheet,
    generate_subject_coherence_diagnostics,
    generate_phase_d_crop_diagnostics,
    generate_visual_review_contact_sheet,
    generate_motion_amplitude_comparison_contact_sheet,
    generate_p0_frame_difference_artifacts,
    generate_phase_1_7_multi_row_contact_sheet,
    generate_visual_review_diagnostics_sheet,
    generate_final_contact_sheet,
    generate_phase_e_keyframe_contact_sheet,
)
from .planner import QualityPlanner
from .hardware import profile_hardware

__all__ = [
    "compute_subject_rigidity_metrics",
    "classify_motion_visibility",
    "evaluate_subject_scale_change",
    "compute_perceptual_motion_score",
    "analyze_image_space_motion_and_subject_fidelity",
    "compute_temporal_diagnostics",
    "export_temporal_motion_profile",
    "export_p0_raster_debug_trace",
    "generate_projected_vs_raster_trajectory_plot",
    "generate_pixel_trajectory_plot",
    "generate_camera_vs_raster_motion_plot",
    "generate_camera_path_plot",
    "generate_layer_displacement_curve_plot",
    "generate_discontinuity_rejection_map",
    "analyze_zero_motion_errors",
    "generate_micro_sweep_contact_sheet",
    "generate_subject_coherence_diagnostics",
    "generate_phase_d_crop_diagnostics",
    "generate_visual_review_contact_sheet",
    "generate_motion_amplitude_comparison_contact_sheet",
    "generate_p0_frame_difference_artifacts",
    "generate_phase_1_7_multi_row_contact_sheet",
    "generate_visual_review_diagnostics_sheet",
    "generate_final_contact_sheet",
    "generate_phase_e_keyframe_contact_sheet",
    "QualityPlanner",
    "profile_hardware",
]
