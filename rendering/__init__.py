"""
Rendering Package Entry Point.
"""

from .disocclusion import reconstruct_background_rgb, complete_background_depth, compute_provenance_map
from .frame_renderer import verify_zero_motion_identity, run_micro_motion_sweep, synthesize_micro_motion_frame, render_phase_e_representative_keyframes
from .sequence_renderer import render_full_frame_sequence
from .video_encoder import verify_ffmpeg, extract_and_verify_mp4_frames, encode_and_verify_mp4

__all__ = [
    "reconstruct_background_rgb",
    "complete_background_depth",
    "compute_provenance_map",
    "verify_zero_motion_identity",
    "run_micro_motion_sweep",
    "synthesize_micro_motion_frame",
    "render_phase_e_representative_keyframes",
    "render_full_frame_sequence",
    "verify_ffmpeg",
    "extract_and_verify_mp4_frames",
    "encode_and_verify_mp4",
]
