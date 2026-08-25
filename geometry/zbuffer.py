import numpy as np
from typing import Tuple

def deterministic_z_buffer_update(
    current_z: np.ndarray,
    current_color: np.ndarray,
    proj_u: np.ndarray,
    proj_v: np.ndarray,
    new_z: np.ndarray,
    new_color: np.ndarray,
    height: int,
    width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic Z-buffer update for forward splatting.
    Closer camera-space Z (smaller Z value) overwrites existing surface.
    """
    z_buf = current_z.copy()
    color_buf = current_color.copy()

    u_int = np.round(proj_u).astype(int)
    v_int = np.round(proj_v).astype(int)

    valid_mask = (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height) & (new_z > 0)

    for i in np.where(valid_mask)[0]:
        x = u_int[i]
        y = v_int[i]
        z_val = new_z[i]
        if z_val < z_buf[y, x]:
            z_buf[y, x] = z_val
            color_buf[y, x] = new_color[i]

    return z_buf, color_buf
