"""
2.5D Diagnostics and Contact Sheet Exporter.
"""

from quality.diagnostics import export_temporal_motion_profile, generate_camera_vs_raster_motion_plot

class Diagnostics25D:
    """Diagnostic generator for Mode A (2.5D)."""

    @staticmethod
    def export_profile(output_dir, frames):
        return export_temporal_motion_profile(output_dir, frames)
