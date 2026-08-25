"""
3D Asset Exporter (OBJ, PLY, GLB).
"""

from render_backend.explicit_3d import export_explicit_3d_assets

class Exporter3D:
    """Exports 3D mesh and point cloud packages."""

    @staticmethod
    def export(rgb_array, depth_map, output_dir):
        return export_explicit_3d_assets(rgb_array, depth_map, output_dir)
