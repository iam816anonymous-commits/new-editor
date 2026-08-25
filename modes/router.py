from typing import Dict, Any
from core.enums import RenderMode

def select_render_mode(requested_mode: str = "auto", motion_style: str = "Cinematic Push-In", motion_strength: str = "Cinematic") -> str:
    req_norm = requested_mode.lower()
    if req_norm == "2.5d":
        return RenderMode.MODE_2_5D.value
    elif req_norm == "3d":
        return RenderMode.MODE_3D.value
    style_upper = motion_style.upper()
    if "3D" in style_upper or "FREE_VIEW" in style_upper or "WIDE_SWEEP" in style_upper:
        return RenderMode.MODE_3D.value
    return RenderMode.MODE_2_5D.value
