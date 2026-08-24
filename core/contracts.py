"""
Typed Pipeline Contracts and Dataclasses.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image

@dataclass
class InputImage:
    path: Path
    pil_image: Image.Image
    rgb_array: np.ndarray
    width: int
    height: int
    full_sha256: str
    short_hash: str

@dataclass
class DepthField:
    raw_depth: np.ndarray
    normalized_depth: np.ndarray
    refined_depth: np.ndarray
    confidence_map: np.ndarray
    depth_min: float
    depth_max: float
    depth_mean: float
    depth_std: float

@dataclass
class SubjectSelectionContract:
    refined_mask: np.ndarray
    dilated_mask: np.ndarray
    boundary_risk_map: np.ndarray
    selection_confidence: float
    score_margin: float
    status: str

@dataclass
class SceneRepresentation:
    rgb_array: np.ndarray
    depth_map: np.ndarray
    background_plate: np.ndarray
    background_depth: np.ndarray
    provenance_map: np.ndarray
    subject_mask: np.ndarray
    boundary_risk_map: np.ndarray

@dataclass
class CameraPose:
    frame_index: int
    translation: np.ndarray  # [tx, ty, tz]
    rotation: np.ndarray     # [pitch, yaw, roll] in radians
    R_matrix: np.ndarray     # 3x3 rotation matrix

@dataclass
class CameraTrajectory:
    style: str
    strength: str
    num_frames: int
    translations: np.ndarray  # (N, 3)
    rotations: np.ndarray     # (N, 3)
    magnitude_scale: float
    disparity_ceiling_px: float

@dataclass
class RenderRequest:
    input_path: Path
    motion_style: str = "Cinematic Push-In"
    motion_strength: str = "Cinematic"
    render_mode: str = "auto"
    quality: str = "auto"
    resolution: str = "auto"
    frame_count: int = 48
    fps: int = 24
    render_video: bool = False
    output_dir: Path = field(default_factory=lambda: Path("output"))

@dataclass
class RenderResult:
    render_id: str
    output_hash_dir: Path
    selected_mode: str
    output_mp4_path: Optional[Path] = None
    metrics_json_path: Optional[Path] = None
    frame_count: int = 0
    duration_seconds: float = 0.0
