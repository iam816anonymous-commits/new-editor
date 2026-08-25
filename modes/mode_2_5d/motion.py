from camera.safety import plan_safe_motion_trajectory, compute_safety_margins

class MotionPlanner25D:
    @staticmethod
    def plan(style, strength, width, height, depth_map, confidence_map, subject_mask, boundary_risk_map, provenance_map, fx, fy, cx, cy, num_frames=48):
        return plan_safe_motion_trajectory(style, strength, width, height, depth_map, confidence_map, subject_mask, boundary_risk_map, provenance_map, fx, fy, cx, cy, num_frames)
