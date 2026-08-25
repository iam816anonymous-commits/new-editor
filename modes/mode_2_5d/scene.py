"""
2.5D Layered Scene Representation.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class Scene25D:
    """Dataclass holding 2.5D layered scene components."""
    rgb_array: np.ndarray
    depth_map: np.ndarray
    subject_mask: np.ndarray
    bg_plate: np.ndarray
    bg_depth: np.ndarray
    confidence_map: np.ndarray
