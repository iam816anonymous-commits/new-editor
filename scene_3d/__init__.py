"""
Isolated 3D Inferred Scene Prototype Module for Phase 2.6.

Provides:
- Inferred3DScene representation orchestrator
- PointCloud3D backprojection and provenance tracking
- MeshGeometry3D depth-to-mesh triangulation
- GaussianScene 3DGS parameter prediction scaffold
- SceneComplexityAnalyzer and ExecutionBackend abstractions
"""

from .camera import PerspectiveCamera3D
from .point_cloud import PointCloud3D, Point3D
from .geometry import MeshGeometry3D, construct_depth_mesh
from .gaussian_scene import GaussianScene, Gaussian3D
from .reconstruction import SceneComplexityTier, SceneComplexityAnalyzer, ExecutionBackend, ExecutionBackendType
from .renderer import Inferred3DRenderer
from .scene import Inferred3DScene, reconstruct_inferred_3d_scene
