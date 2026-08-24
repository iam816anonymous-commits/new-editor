"""
Scene Complexity Router and Adaptive Execution Backend Abstraction for scene_3d/.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import torch


class SceneComplexityTier(str, Enum):
    SIMPLE = "SIMPLE"
    MODERATE = "MODERATE"
    COMPLEX = "COMPLEX"


class ExecutionBackendType(str, Enum):
    CPU = "CPU"
    CUDA = "CUDA"
    HYBRID = "HYBRID"


@dataclass
class ExecutionBackend:
    backend_type: ExecutionBackendType
    device: torch.device
    precision: torch.dtype = torch.float32

    @classmethod
    def auto_select(cls) -> 'ExecutionBackend':
        if torch.cuda.is_available():
            return cls(backend_type=ExecutionBackendType.CUDA, device=torch.device("cuda"))
        return cls(backend_type=ExecutionBackendType.CPU, device=torch.device("cpu"))


class SceneComplexityAnalyzer:
    """
    Evaluates image-space signals (depth variance, subject count, edge density, disocclusion risk)
    to classify scene reconstruction difficulty tier.
    """

    @staticmethod
    def analyze(
        rgb_array: np.ndarray,
        depth_map: np.ndarray,
        subject_mask: np.ndarray
    ) -> Tuple[SceneComplexityTier, float, Dict[str, Any]]:
        depth_std = float(np.std(depth_map))
        sub_area_pct = float(np.sum(subject_mask) / max(1, subject_mask.size))

        # Compute boundary edge density
        sub_uint = (subject_mask * 255).astype(np.uint8)
        contours, _ = cv2_find_contours(sub_uint)
        peri = float(sum(len(c) for c in contours))
        edge_density = peri / max(1.0, float(np.sum(subject_mask)))

        score = 0.4 * min(1.0, depth_std / 2.0) + 0.3 * sub_area_pct + 0.3 * min(1.0, edge_density * 5.0)

        if score >= 0.70:
            tier = SceneComplexityTier.COMPLEX
        elif score >= 0.35:
            tier = SceneComplexityTier.MODERATE
        else:
            tier = SceneComplexityTier.SIMPLE

        details = {
            "complexity_score": round(score, 4),
            "depth_std": round(depth_std, 4),
            "subject_area_pct": round(sub_area_pct, 4),
            "edge_density": round(edge_density, 4)
        }
        return tier, score, details


def cv2_find_contours(mask_uint8: np.ndarray):
    import cv2
    return cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
