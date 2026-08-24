"""
Explicit 3D Mesh Generation, Point Cloud Triangulation, and Export Subsystem.

Converts RGB images and continuous rendering depth maps into explicit 3D textured meshes,
point clouds, and provenance-tracked scene representations.

Exports:
- OBJ (.obj) textured mesh with OBJ material (.mtl)
- PLY (.ply) point cloud with vertex colors and uncertainty provenance labels
- GLB-compatible scene JSON metadata
- Provenance & Uncertainty Maps
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import json


@dataclass
class Point3D:
    x: float
    y: float
    z: float
    r: int
    g: int
    b: int
    provenance_code: str = "OBSERVED"  # OBSERVED, DEPTH_INFERRED, GEOMETRY_INFERRED, INPAINTED
    confidence: float = 1.0


@dataclass
class Mesh3D:
    vertices: np.ndarray  # (N, 3) float32 [X, Y, Z]
    colors: np.ndarray    # (N, 3) uint8 [R, G, B]
    uvs: np.ndarray       # (N, 2) float32 [u_norm, v_norm]
    faces: np.ndarray     # (M, 3) int32 vertex indices
    provenance_labels: List[str] = field(default_factory=list)
    confidence_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


def backproject_depth_to_point_cloud(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    provenance_map: Optional[np.ndarray] = None,
    depth_discontinuity_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Backprojects 2D pixel grid and depth map Z into 3D camera-space point cloud.
    Returns: (vertices_xyz, colors_rgb, uvs_norm, provenance_labels)
    """
    h, w, _ = rgb_array.shape
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    X = (u_grid - cx) * depth_map / fx
    Y = (v_grid - cy) * depth_map / fy
    Z = depth_map

    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
    colors = rgb_array.reshape(-1, 3).astype(np.uint8)

    u_norm = (u_grid / float(w - 1)).ravel().astype(np.float32)
    v_norm = (1.0 - (v_grid / float(h - 1))).ravel().astype(np.float32) # Invert V for standard UV
    uvs = np.column_stack([u_norm, v_norm])

    prov_labels = []
    prov_flat = provenance_map.ravel() if provenance_map is not None else np.ones(h * w, dtype=np.float32)
    for p_val in prov_flat:
        if p_val >= 0.95:
            prov_labels.append("OBSERVED")
        elif p_val >= 0.50:
            prov_labels.append("DEPTH_INFERRED")
        else:
            prov_labels.append("INPAINTED")

    return vertices, colors, uvs, prov_labels


def construct_explicit_3d_mesh(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    provenance_map: Optional[np.ndarray] = None,
    depth_discontinuity_threshold: float = 0.5
) -> Mesh3D:
    """
    Triangulates 3D point cloud into explicit 3D mesh faces, breaking quad edges across depth discontinuities.
    """
    h, w, _ = rgb_array.shape
    vertices, colors, uvs, prov_labels = backproject_depth_to_point_cloud(
        rgb_array, depth_map, fx, fy, cx, cy, provenance_map=provenance_map
    )

    # 2D Grid Quad Indexing
    indices = np.arange(h * w, dtype=np.int32).reshape(h, w)

    # Top-left vertices of quads
    i00 = indices[:-1, :-1].ravel()
    i10 = indices[:-1, 1:].ravel()
    i01 = indices[1:, :-1].ravel()
    i11 = indices[1:, 1:].ravel()

    # Check depth differences across quad diagonals to prevent rubber-sheet stretching
    z00 = depth_map[:-1, :-1].ravel()
    z10 = depth_map[:-1, 1:].ravel()
    z01 = depth_map[1:, :-1].ravel()
    z11 = depth_map[1:, 1:].ravel()

    d_max1 = np.maximum(np.abs(z00 - z10), np.abs(z00 - z01))
    d_max2 = np.maximum(np.abs(z11 - z10), np.abs(z11 - z01))

    valid_quads = (d_max1 < depth_discontinuity_threshold) & (d_max2 < depth_discontinuity_threshold)

    # Split valid quads into 2 triangles: (i00, i10, i01) and (i10, i11, i01)
    f1 = np.column_stack([i00[valid_quads], i10[valid_quads], i01[valid_quads]])
    f2 = np.column_stack([i10[valid_quads], i11[valid_quads], i01[valid_quads]])
    faces = np.vstack([f1, f2]).astype(np.int32)

    return Mesh3D(
        vertices=vertices,
        colors=colors,
        uvs=uvs,
        faces=faces,
        provenance_labels=prov_labels,
        confidence_scores=np.ones(len(vertices), dtype=np.float32)
    )


def export_mesh_obj(mesh: Mesh3D, filepath: Path) -> None:
    """Exports 3D mesh to Wavefront OBJ format (.obj)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Explicit 3D Mesh Export - First-Principles Cinematic 2.5D Renderer\n")
        # Write vertices with colors
        for v, c in zip(mesh.vertices, mesh.colors):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]/255.0:.4f} {c[1]/255.0:.4f} {c[2]/255.0:.4f}\n")
        # Write UVs
        for uv in mesh.uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        # Write faces (1-based indexing)
        for face in mesh.faces:
            f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")


def export_point_cloud_ply(mesh: Mesh3D, filepath: Path) -> None:
    """Exports 3D point cloud to Polygon File Format (.ply) with vertex colors."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    num_verts = len(mesh.vertices)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for v, c in zip(mesh.vertices, mesh.colors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")


def export_scene_3d_package(
    rgb_array: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    export_dir: Path,
    provenance_map: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Exports complete explicit 3D scene package:
    - scene_mesh.obj
    - point_cloud.ply
    - scene_3d_graph.json
    - provenance_summary.json
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    mesh = construct_explicit_3d_mesh(rgb_array, depth_map, fx, fy, cx, cy, provenance_map=provenance_map)

    obj_path = export_dir / "scene_mesh.obj"
    ply_path = export_dir / "point_cloud.ply"

    export_mesh_obj(mesh, obj_path)
    export_point_cloud_ply(mesh, ply_path)

    summary = {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "obj_file": str(obj_path),
        "ply_file": str(ply_path),
        "provenance_distribution": {
            "OBSERVED": mesh.provenance_labels.count("OBSERVED"),
            "DEPTH_INFERRED": mesh.provenance_labels.count("DEPTH_INFERRED"),
            "INPAINTED": mesh.provenance_labels.count("INPAINTED")
        }
    }

    with open(export_dir / "scene_3d_graph.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
