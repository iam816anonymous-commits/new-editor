import sys
import cv2
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Any, Optional, Tuple

def verify_ffmpeg() -> str:
    """Verifies that FFmpeg is available on system PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else "ffmpeg found"
        return first_line
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        sys.stderr.write("ERROR: FFmpeg is not installed or not found on system PATH.\n")
        sys.stderr.write("Please install FFmpeg to proceed with video generation.\n")
        raise RuntimeError("FFmpeg verification failed.") from e

def extract_and_verify_mp4_frames(
    output_mp4_path: Path,
    rendered_frames: list,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Extracts keyframes (video_frame_00.png, video_frame_24.png, video_frame_47.png) from encoded MP4
    and compares them against rendered PNG source frames to verify FFmpeg encoding fidelity.
    """
    cap = cv2.VideoCapture(str(output_mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for frame extraction: {output_mp4_path}")

    extracted_metrics = {}
    sample_indices = [0, 24, 47]

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to extract frame {idx} from MP4 {output_mp4_path}")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ext_path = output_dir / f"video_frame_{idx:02d}.png"
        Image.fromarray(frame_rgb).save(ext_path)

        src_rgb = rendered_frames[idx]
        abs_diff = np.abs(frame_rgb.astype(np.float32) - src_rgb.astype(np.float32))
        mae = float(np.mean(abs_diff))
        rmse = float(np.sqrt(np.mean(abs_diff ** 2)))
        max_diff = float(np.max(abs_diff))

        extracted_metrics[f"frame_{idx:02d}"] = {
            "extracted_path": str(ext_path),
            "mae_vs_source_png": mae,
            "rmse_vs_source_png": rmse,
            "max_pixel_diff": max_diff
        }

    cap.release()
    return extracted_metrics

def encode_and_verify_mp4(
    frames_dir: Path,
    output_mp4_path: Path,
    fps: int = 24,
    expected_frames: int = 48,
    expected_resolution: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Encodes generated PNG frames in frames_dir to output_mp4_path using FFmpeg at fps=24 with libx264 high quality (crf=17).
    Validates output MP4 via OpenCV VideoCapture verifying actual_frames == expected_frames, FPS, resolution, and duration.
    Raises RuntimeError if frame count mismatches expected_frames.
    Returns video metadata dictionary.
    """
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    # Support 4-digit or 2-digit zero padded frame filenames
    if (frames_dir / "frame_0000.png").exists():
        input_pattern = str(frames_dir / "frame_%04d.png")
    else:
        input_pattern = str(frames_dir / "frame_%02d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "17",
        str(output_mp4_path)
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg encoding failed for '{output_mp4_path}': {e.stderr.decode()}") from e

    if not output_mp4_path.exists() or output_mp4_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg output file '{output_mp4_path}' is missing or empty.")

    # Verify metadata using OpenCV VideoCapture
    cap = cv2.VideoCapture(str(output_mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open generated MP4 video at '{output_mp4_path}' using OpenCV.")

    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_sec = actual_frames / max(actual_fps, 1e-3)

    if actual_frames != expected_frames:
        raise ValueError(f"MP4 frame count mismatch: expected {expected_frames}, got {actual_frames}")
    if abs(actual_fps - fps) > 0.5:
        raise ValueError(f"MP4 FPS mismatch: expected {fps}, got {actual_fps}")
    if expected_resolution is not None and (width, height) != expected_resolution:
        raise ValueError(f"MP4 resolution mismatch: expected {expected_resolution}, got ({width}, {height})")

    return {
        "mp4_file": str(output_mp4_path),
        "file_size_bytes": output_mp4_path.stat().st_size,
        "frame_count": actual_frames,
        "fps": actual_fps,
        "width": width,
        "height": height,
        "duration_seconds": duration_sec
    }
