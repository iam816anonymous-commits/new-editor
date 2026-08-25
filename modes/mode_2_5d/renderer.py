import numpy as np
from geometry.splatting import render_single_frame_forward_splatting

class Renderer25D:
    @staticmethod
    def render_frame(rgb_array, depth_map, bg_plate, bg_depth, provenance_map, R, t, fx, fy, cx, cy):
        return render_single_frame_forward_splatting(rgb_array, depth_map, bg_plate, bg_depth, provenance_map, R, t, fx, fy, cx, cy)
