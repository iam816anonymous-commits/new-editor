"""
Output Package Entry Point.
"""

from .artifacts import (
    compute_image_sha256,
    validate_and_load_image,
    setup_cache_directory,
    setup_output_directories,
    save_phase_b_diagnostic_artifacts,
    save_phase_c_diagnostic_artifacts,
    save_phase_d_validation_artifacts,
    save_phase_d_diagnostic_artifacts,
    save_phase_e_artifacts,
    validate_output_contract,
)
from .video import verify_video_output
from .manifests import export_render_manifest

__all__ = [
    "compute_image_sha256",
    "validate_and_load_image",
    "setup_cache_directory",
    "setup_output_directories",
    "save_phase_b_diagnostic_artifacts",
    "save_phase_c_diagnostic_artifacts",
    "save_phase_d_validation_artifacts",
    "save_phase_d_diagnostic_artifacts",
    "save_phase_e_artifacts",
    "validate_output_contract",
    "verify_video_output",
    "export_render_manifest",
]
