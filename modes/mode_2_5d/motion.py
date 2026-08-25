"""
2.5D Closed-Loop Trajectory Safety Planner.
"""

from camera.safety import plan_safe_motion_trajectory, compute_safety_margins

class MotionPlanner25D:
    """Motion planning wrapper for 2.5D trajectories."""

    @staticmethod
    def plan(depth_map, subject_mask, motion_style, motion_strength, frame_count):
        return plan_safe_motion_trajectory(depth_map, subject_mask, motion_style, motion_strength, frame_count)
