from quality import export_temporal_motion_profile, generate_camera_vs_raster_motion_plot

class Diagnostics25D:
    @staticmethod
    def export_profile(output_dir, frames):
        return export_temporal_motion_profile(output_dir, frames)
