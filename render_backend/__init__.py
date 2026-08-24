"""
Modular Render Backend Package for First-Principles Cinematic Renderer.

Supports:
- 2.5D Layered Parallax Backend (Default Reference)
- Explicit 3D Mesh Reconstruction Backend (OBJ/PLY/GLB Export)
- 3D Gaussian Splatting Scaffold Backend
- Hybrid Mesh + 3DGS Backend
"""

from .explicit_3d import (
    Point3D,
    Mesh3D,
    construct_explicit_3d_mesh,
    export_mesh_obj,
    export_point_cloud_ply,
    export_scene_3d_package
)
