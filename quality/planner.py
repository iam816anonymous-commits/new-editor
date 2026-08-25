from typing import Dict, Any, Optional
from core.enums import QualityTier

class QualityPlanner:
    @staticmethod
    def plan_quality_profile(requested_quality: str = "auto", requested_resolution: str = "auto", device: str = "cpu") -> Dict[str, Any]:
        resolved_quality = requested_quality.lower()
        if resolved_quality == "auto":
            resolved_quality = "balanced" if device == "cuda" else "fast"
        resolved_res = requested_resolution.lower()
        if resolved_res == "auto":
            target_res = (1024, 683) if device == "cuda" else (640, 360)
        elif resolved_res == "720p":
            target_res = (1280, 720)
        elif resolved_res == "1080p":
            target_res = (1920, 1080)
        else:
            target_res = (1024, 683)
        return {
            "requested_quality": requested_quality,
            "resolved_quality": resolved_quality,
            "target_resolution": target_res,
            "device": device,
            "cpu_ceiling_enforced": device == "cpu" and target_res[1] > 720
        }
