import numpy as np
from scene_3d.scene import Inferred3DScene

class Renderer3D:
    @staticmethod
    def render(scene: Inferred3DScene, R_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
        return scene.render_novel_view(R_mat, t_vec)
