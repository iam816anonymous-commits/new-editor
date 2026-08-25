from render_backend.explicit_3d import export_scene_3d_package

class Exporter3D:
    @staticmethod
    def export(rgb_array, depth_map, output_dir):
        return export_scene_3d_package(rgb_array, depth_map, output_dir)
