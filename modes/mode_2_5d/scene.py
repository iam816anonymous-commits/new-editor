import numpy as np
from dataclasses import dataclass

@dataclass
class Scene25D:
    rgb_array: np.ndarray
    depth_map: np.ndarray
    subject_mask: np.ndarray
    bg_plate: np.ndarray
    bg_depth: np.ndarray
    confidence_map: np.ndarray
