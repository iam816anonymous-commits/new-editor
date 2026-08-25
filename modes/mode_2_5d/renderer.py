import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .scene import LayeredScene, SceneLayer
from geometry.splatting import render_single_frame_forward_splatting
from geometry.transforms import regularize_subject_depth, compute_subject_rigid_transform

class Renderer25D:
    @staticmethod
    def render_frame(rgb_array, depth_map, bg_plate, bg_depth, provenance_map, R, t, fx, fy, cx, cy):
        return render_single_frame_forward_splatting(rgb_array, depth_map, bg_plate, bg_depth, provenance_map, R, t, fx, fy, cx, cy)

    @staticmethod
    def render_layered_scene(
        layered_scene: LayeredScene,
        R: np.ndarray,
        t: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Renders a multi-layer 2.5D scene frame by composition over depth priority.
        Applies subject depth regularization and rigid SE(3) transformation to primary subject.
        Returns: (rendered_rgb, rendered_z_buffer, rendered_provenance)
        """
        h, w = layered_scene.source_image.shape[:2]
        accum_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        accum_z = np.full((h, w), 1e6, dtype=np.float32)
        accum_prov = np.ones((h, w), dtype=np.float32)

        # Sort layers by priority ascending (background to foreground)
        sorted_layers = sorted(layered_scene.layers, key=lambda l: l.priority)

        bg_plate = layered_scene.source_image
        bg_depth = layered_scene.refined_depth

        for layer in sorted_layers:
            layer_prov = np.ones((h, w), dtype=np.float32)
            layer_prov[layer.inpaint_mask] = 0.0

            if layer.semantic_role == "PRIMARY_SUBJECT":
                # Regularize depth to prevent independent pixel swimming
                layer_depth = regularize_subject_depth(
                    layer.depth, layer.valid_mask, guide_image=layer.rgb
                )
                # Compute subject-locked rigid SE(3) transformation
                ref_d = layer.anchor.reference_depth if layer.anchor else float(np.median(layer_depth[layer.valid_mask]))
                R_layer, t_layer = compute_subject_rigid_transform(R, t, ref_d)
            else:
                layer_depth = layer.depth
                R_layer, t_layer = R, t

            s_rgb, s_z, s_prov = render_single_frame_forward_splatting(
                layer.rgb, layer_depth, bg_plate, bg_depth, layer_prov,
                R_layer, t_layer, fx, fy, cx, cy
            )

            valid_pixels = s_z < accum_z
            accum_rgb[valid_pixels] = s_rgb[valid_pixels]
            accum_z[valid_pixels] = s_z[valid_pixels]
            accum_prov[valid_pixels] = s_prov[valid_pixels]

        return accum_rgb, accum_z, accum_prov
