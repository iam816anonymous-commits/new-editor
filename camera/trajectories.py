import numpy as np
from typing import Tuple, Optional

def generate_c1_smooth_trajectory(
    style: str,
    magnitude_scale: float,
    num_frames: int = 48,
    is_loop: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates camera trajectory poses for t in [0, 1].

    Semantics Separation:
    - Non-looping trajectories (e.g. CINEMATIC_PUSH_IN when is_loop is False or default):
      Monotonic progressive camera movement toward target with smooth C1 acceleration/deceleration.
      P(0) != P(1), V(0) = V(1) = 0.
    - Looping trajectories (e.g. ORBIT, MICRO_ORBIT, or explicit CINEMATIC_LOOP / is_loop=True):
      Smooth closed trajectory satisfying position closure P(0) == P(1) and velocity closure V(0) == V(1) == 0.

    Returns:
    - translations: (num_frames, 3) array [tx, ty, tz]
    - rotations: (num_frames, 3) array [pitch, yaw, roll] in radians
    """
    t = np.linspace(0.0, 1.0, num_frames, endpoint=True)
    w_loop = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))

    translations = np.zeros((num_frames, 3), dtype=np.float64)
    rotations = np.zeros((num_frames, 3), dtype=np.float64)

    style_upper = style.upper().replace(" ", "_").replace("-", "_")

    # Quintic Smoothstep Easing s(t) = 6t^5 - 15t^4 + 10t^3 (smooth C1 velocity at t=0 and t=1, s(0)=0, s(1)=1)
    s_quintic = 6.0 * (t ** 5) - 15.0 * (t ** 4) + 10.0 * (t ** 3)

    if is_loop is True or style_upper in ["CINEMATIC_LOOP", "LOOP"]:
        # Forced looping trajectory
        translations[:, 2] = w_loop * s_quintic * magnitude_scale * 0.22
        translations[:, 1] = -w_loop * s_quintic * magnitude_scale * 0.025
        translations[:, 0] = w_loop * np.sin(2.0 * np.pi * t) * magnitude_scale * 0.012
        rotations[:, 0] = -w_loop * s_quintic * magnitude_scale * np.radians(1.2)
    elif style_upper == "STATIC":
        pass  # All zeros
    elif style_upper in ["SUBTLE_PUSH_IN", "CINEMATIC_PUSH_IN", "CINEMATIC_PUSHIN", "PUSH_IN", "PUSHIN"]:
        # Progressive Push-In: smooth camera travel toward scene with progressive lateral & vertical move
        translations[:, 2] = -s_quintic * magnitude_scale * 0.25
        translations[:, 1] = -s_quintic * magnitude_scale * 0.04
        translations[:, 0] = s_quintic * magnitude_scale * 0.08
        rotations[:, 0] = -s_quintic * magnitude_scale * np.radians(0.8)
        rotations[:, 1] = s_quintic * magnitude_scale * np.radians(0.6)
    elif style_upper in ["SLOW_DOLLY_LEFT", "DOLLY_LEFT", "PAN_LEFT", "PANLEFT"]:
        translations[:, 0] = -s_quintic * magnitude_scale * 0.12
        translations[:, 2] = -s_quintic * magnitude_scale * 0.05
        rotations[:, 1] = -s_quintic * magnitude_scale * np.radians(1.2)
    elif style_upper in ["SLOW_DOLLY_RIGHT", "DOLLY_RIGHT", "PAN_RIGHT", "PANRIGHT"]:
        translations[:, 0] = s_quintic * magnitude_scale * 0.12
        translations[:, 2] = -s_quintic * magnitude_scale * 0.05
        rotations[:, 1] = s_quintic * magnitude_scale * np.radians(1.2)
    elif style_upper in ["VERTICAL_DRIFT", "PAN_UP", "PAN_DOWN", "VERTICAL_PAN"]:
        translations[:, 1] = -s_quintic * magnitude_scale * 0.10
        translations[:, 2] = -s_quintic * magnitude_scale * 0.05
        rotations[:, 0] = -s_quintic * magnitude_scale * np.radians(1.0)
    elif style_upper in ["DIAGONAL_DOLLY"]:
        translations[:, 0] = s_quintic * magnitude_scale * 0.10
        translations[:, 1] = -s_quintic * magnitude_scale * 0.08
        translations[:, 2] = -s_quintic * magnitude_scale * 0.10
        rotations[:, 0] = -s_quintic * magnitude_scale * np.radians(0.6)
        rotations[:, 1] = s_quintic * magnitude_scale * np.radians(0.8)
    elif style_upper in ["PARALLAX_PUSH"]:
        translations[:, 0] = s_quintic * magnitude_scale * 0.12
        translations[:, 1] = -s_quintic * magnitude_scale * 0.06
        translations[:, 2] = -s_quintic * magnitude_scale * 0.25
        rotations[:, 0] = -s_quintic * magnitude_scale * np.radians(0.8)
        rotations[:, 1] = s_quintic * magnitude_scale * np.radians(1.0)
    elif style_upper in ["DOLLY_IN", "DOLLYIN"]:
        translations[:, 2] = -s_quintic * magnitude_scale * 0.20
    elif style_upper in ["DOLLY_OUT", "DOLLYOUT"]:
        translations[:, 2] = s_quintic * magnitude_scale * 0.20
    elif style_upper in ["CONTROL_50PX", "CONTROL_100PX", "CONTROL_200PX", "CONTROL_400PX"]:
        px_targets = {"CONTROL_50PX": 50.0, "CONTROL_100PX": 100.0, "CONTROL_200PX": 200.0, "CONTROL_400PX": 400.0}
        target_shift = px_targets[style_upper] * magnitude_scale
        # For Z=5.0 and fx=320, tx = target_shift * Z / fx
        tx_calc = (target_shift * 5.0) / 320.0
        translations[:, 0] = s_quintic * tx_calc
    elif style_upper in ["ORBIT", "MICRO_ORBIT"]:
        scale_t = 0.03 if style_upper == "MICRO_ORBIT" else 0.05
        # Modulate orbit coordinates with C1 window w_loop(t) so velocity starts and ends strictly at 0
        translations[:, 0] = w_loop * np.sin(2.0 * np.pi * t) * magnitude_scale * scale_t
        translations[:, 1] = w_loop * (np.cos(2.0 * np.pi * t) - 1.0) * magnitude_scale * (scale_t * 0.5)
        rotations[:, 1] = w_loop * np.sin(2.0 * np.pi * t) * magnitude_scale * np.radians(1.5)
        rotations[:, 0] = -w_loop * (np.cos(2.0 * np.pi * t) - 1.0) * magnitude_scale * np.radians(1.0)
    else:
        translations[:, 0] = w_loop * np.sin(2.0 * np.pi * t) * magnitude_scale * 0.04
        rotations[:, 1] = w_loop * np.sin(2.0 * np.pi * t) * magnitude_scale * np.radians(1.5)

    return translations, rotations
