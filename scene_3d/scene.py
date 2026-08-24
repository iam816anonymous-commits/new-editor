"""
Main Inferred 3D Scene Representation Orchestrator for scene_3d/.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import json

from .camera import PerspectiveCamera3D
from .point_cloud import PointCloud3D
from .geometry import MeshGeometry3D, construct_depth_mesh
from .gaussian_scene import GaussianScene
from .reconstruction import SceneComplexityAnalyzer, SceneComplexityTier, ExecutionBackend
from .renderer import Inferred3DRenderer


@dataclass
class Inferred3DScene:
    camera: PerspectiveCamera3D
    point_cloud: PointCloud3D
    mesh: MeshGeometry3D
    gaussian_scene: GaussianScene
    complexity_tier: SceneComplexityTier
    complexity_score: float
    execution_backend: ExecutionBackend

    def render_novel_view(self, R_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
        syn_rgb, _ = Inferred3DRenderer.render_point_cloud_view(self.point_cloud, self.camera, R_mat, t_vec)
        return syn_rgb

    def export_3d_package(self, export_dir: Path) -> Dict[str, Any]:
        export_dir.mkdir(parents=True, exist_ok=True)
        from render_backend.explicit_3d import export_scene_3d_package
        summary = export_scene_3d_package(
            self.point_cloud.colors.reshape(self.camera.height, self.camera.width, 3),
            self.point_cloud.vertices[:, 2].reshape(self.camera.height, self.camera.width),
            self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy,
            export_dir
        )
        summary["complexity_tier"] = self.complexity_tier.value
        summary["complexity_score"] = self.complexity_score
        summary["backend"] = self.execution_backend.backend_type.value

        with open(export_dir / "inferred_3d_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


def reconstruct_inferred_3d_scene(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    subject_mask: np.ndarray,
    provenance_map: Optional[np.ndarray] = None,
    backend: Optional[ExecutionBackend] = None
) -> Inferred3DScene:
    h, w, _ = rgb_array.shape
    camera = PerspectiveCamera3D.create_canonical(w, h)

    tier, score, _ = SceneComplexityAnalyzer.analyze(rgb_array, depth_map, subject_mask)
    selected_backend = backend if backend is not None else ExecutionBackend.auto_select()

    pt_cloud = PointCloud3D.from_rgb_depth(rgb_array, depth_map, camera, provenance_map=provenance_map)
    mesh = construct_depth_mesh(rgb_array, depth_map, camera, provenance_map=provenance_map)
    gaussians = GaussianScene.from_point_cloud(pt_cloud)

    return Inferred3DScene(
        camera=camera,
        point_cloud=pt_cloud,
        mesh=mesh,
        gaussian_scene=gaussians,
        complexity_tier=tier,
        complexity_score=score,
        execution_backend=selected_backend
    )
