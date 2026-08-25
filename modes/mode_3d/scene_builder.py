from scene_3d.scene import reconstruct_inferred_3d_scene, Inferred3DScene

class Inferred3DSceneBuilder:
    @staticmethod
    def build(rgb_array, depth_map, subject_mask):
        return reconstruct_inferred_3d_scene(rgb_array=rgb_array, depth_map=depth_map, subject_mask=subject_mask)
