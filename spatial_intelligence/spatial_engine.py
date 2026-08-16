"""
Main Orchestrator for Spatial Intelligence Subsystem.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import cv2
import numpy as np
from PIL import Image

from .schemas import (
    Entity,
    EntityPart,
    SceneGraph,
    DepthField,
    OcclusionRelationship,
    CameraModel,
    SpatialConfidence,
    SpatialDiagnostics
)
from .scene_graph import create_scene_graph, export_scene_graph_dict
from .relationship_inferencer import infer_spatial_relationships
from .depth_field import build_spatial_depth_field
from .occlusion_model import build_occlusion_model, generate_occlusion_map
from .camera_model import create_perspective_camera, export_camera_path_dict, compute_layer_disparity
from .temporal_tracker import TemporalSpatialTracker


def extract_scene_entities(
    subject_selection_result: Any,
    refined_depth: np.ndarray,
    rgb_array: np.ndarray
) -> List[Entity]:
    """
    Extracts structured Entity and EntityPart instances from subject selection candidates and scene depth.
    Converts compound primary subject group into primary subject Entity with sub-parts, and non-selected candidates into background/environmental Entities.
    """
    h, w = refined_depth.shape
    total_pixels = max(1, h * w)
    entities: List[Entity] = []

    features_list = subject_selection_result.candidate_features_list
    selected_group_ids = subject_selection_result.selected_group_ids
    primary_mask = subject_selection_result.refined_mask

    # 1. Primary Compound Subject Entity
    sub_y, sub_x = np.where(primary_mask)
    if len(sub_y) > 0:
        ymin, ymax = int(np.min(sub_y)), int(np.max(sub_y))
        xmin, xmax = int(np.min(sub_x)), int(np.max(sub_x))
        sub_bbox = (ymin, xmin, ymax, xmax)
        sub_cy, sub_cx = float(np.mean(sub_y)), float(np.mean(sub_x))
    else:
        sub_bbox = (0, 0, h - 1, w - 1)
        sub_cy, sub_cx = h / 2.0, w / 2.0

    sub_depths = refined_depth[primary_mask] if np.any(primary_mask) else np.array([5.0])
    primary_area = int(np.sum(primary_mask))

    # Extract sub-parts for primary compound subject
    parts: List[EntityPart] = []
    for f in features_list:
        if f.candidate_id in selected_group_ids:
            # Re-create boolean mask for candidate from features bbox/centroid region or candidate features
            p_y0, p_x0, p_y1, p_x1 = f.bbox
            part_mask = np.zeros((h, w), dtype=bool)
            part_mask[p_y0:p_y1+1, p_x0:p_x1+1] = primary_mask[p_y0:p_y1+1, p_x0:p_x1+1]

            parts.append(EntityPart(
                part_id=f.candidate_id,
                entity_id=1,
                name=f"primary_part_{f.candidate_id}",
                mask=part_mask,
                bbox=f.bbox,
                depth_mean=f.foreground_depth_mean,
                depth_std=f.foreground_depth_std,
                confidence=f.sam_confidence
            ))

    primary_entity = Entity(
        entity_id=1,
        name="primary_compound_subject",
        mask=primary_mask,
        bbox=sub_bbox,
        centroid=(sub_cy, sub_cx),
        norm_centroid=(sub_cy / max(1, h), sub_cx / max(1, w)),
        area_pixels=primary_area,
        area_ratio=float(primary_area / total_pixels),
        depth_mean=float(np.mean(sub_depths)),
        depth_median=float(np.median(sub_depths)),
        depth_std=float(np.std(sub_depths)),
        is_primary_subject=True,
        parts=parts
    )
    entities.append(primary_entity)

    # 2. Secondary/Environmental Entities from non-selected candidate features
    ent_id_counter = 2
    for f in features_list:
        if f.candidate_id not in selected_group_ids and f.mask_area_ratio >= 0.01:
            p_y0, p_x0, p_y1, p_x1 = f.bbox
            sec_mask = np.zeros((h, w), dtype=bool)
            sec_mask[p_y0:p_y1+1, p_x0:p_x1+1] = True
            sec_mask &= (~primary_mask)  # Ensure non-overlapping with primary subject

            if np.sum(sec_mask) > 100:
                sec_depths = refined_depth[sec_mask]
                entities.append(Entity(
                    entity_id=ent_id_counter,
                    name=f"environmental_entity_{f.candidate_id}",
                    mask=sec_mask,
                    bbox=f.bbox,
                    centroid=(f.centroid_y, f.centroid_x),
                    norm_centroid=(f.norm_centroid_y, f.norm_centroid_x),
                    area_pixels=int(np.sum(sec_mask)),
                    area_ratio=float(np.sum(sec_mask) / total_pixels),
                    depth_mean=float(np.mean(sec_depths)),
                    depth_median=float(np.median(sec_depths)),
                    depth_std=float(np.std(sec_depths)),
                    is_primary_subject=False,
                    parts=[]
                ))
                ent_id_counter += 1

    return entities


def export_spatial_diagnostics_artifacts(
    hash_dir: Path,
    original_rgb: np.ndarray,
    diagnostics: SpatialDiagnostics
) -> None:
    """Exports all spatial diagnostics artifacts and JSON files to hash_dir."""
    hash_dir.mkdir(parents=True, exist_ok=True)
    h, w, _ = original_rgb.shape

    # 1. spatial_scene.json
    scene_dict = export_scene_graph_dict(diagnostics.scene_graph)
    with open(hash_dir / "spatial_scene.json", "w") as f:
        json.dump(scene_dict, f, indent=2)

    # 2. spatial_relationships.json
    rel_dict = [
        {
            "subject_id": r.subject_id,
            "target_id": r.target_id,
            "relation": r.relation_type.value,
            "confidence": r.confidence,
            "evidence": r.evidence
        }
        for r in diagnostics.scene_graph.relationships
    ]
    with open(hash_dir / "spatial_relationships.json", "w") as f:
        json.dump(rel_dict, f, indent=2)

    # 3. depth_field.png
    d_field = diagnostics.depth_field
    d_vis = ((d_field.rendering_depth - d_field.min_depth) / max(1e-5, d_field.max_depth - d_field.min_depth) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(d_vis).save(hash_dir / "depth_field.png")

    # 4. depth_uncertainty.png
    unc_vis = (d_field.uncertainty_map * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(unc_vis).save(hash_dir / "depth_uncertainty.png")

    # 5. occlusion_map.png
    occ_map = generate_occlusion_map(diagnostics.occlusion_relationships, (h, w))
    Image.fromarray(occ_map).save(hash_dir / "occlusion_map.png")

    # 6. camera_path.json
    dummy_t = np.zeros((48, 3))
    dummy_r = np.zeros((48, 3))
    cam_dict = export_camera_path_dict(diagnostics.camera_model, dummy_t, dummy_r)
    with open(hash_dir / "camera_path.json", "w") as f:
        json.dump(cam_dict, f, indent=2)

    # 7. spatial_diagnostics.json
    diag_summary = {
        "spatial_confidence": {
            "relationship_confidence": diagnostics.spatial_confidence.relationship_confidence,
            "depth_field_confidence": diagnostics.spatial_confidence.depth_field_confidence,
            "occlusion_confidence": diagnostics.spatial_confidence.occlusion_confidence,
            "overall_spatial_confidence": diagnostics.spatial_confidence.overall_spatial_confidence
        },
        "metrics_summary": diagnostics.metrics_summary
    }
    with open(hash_dir / "spatial_diagnostics.json", "w") as f:
        json.dump(diag_summary, f, indent=2)


def analyze_spatial_scene(
    rgb_array: np.ndarray,
    refined_depth: np.ndarray,
    background_depth: np.ndarray,
    confidence_map: np.ndarray,
    provenance_map: np.ndarray,
    subject_selection_result: Any,
    hash_dir: Optional[Path] = None
) -> SpatialDiagnostics:
    """
    Main entry point for Spatial Intelligence subsystem.
    Executes scene entity extraction, scene graph construction, spatial relationship inference,
    depth field building, occlusion modeling, camera model creation, and spatial diagnostics export.
    """
    h, w, _ = rgb_array.shape

    # 1. Extract Entities & Parts
    entities = extract_scene_entities(subject_selection_result, refined_depth, rgb_array)

    # 2. Build Scene Graph & Infer Relationships
    scene_graph = create_scene_graph(entities)
    scene_graph = infer_spatial_relationships(scene_graph, (h, w))

    # 3. Build Structured 2.5D Depth Field
    depth_field = build_spatial_depth_field(
        refined_depth, background_depth, subject_selection_result.refined_mask,
        confidence_map, provenance_map, scene_graph
    )

    # 4. Build Occlusion Model
    occlusion_rels = build_occlusion_model(scene_graph, (h, w))

    # 5. Create Camera Model
    camera_model = create_perspective_camera(w, h)

    # 6. Calculate Spatial Confidence
    rel_conf = float(np.mean([r.confidence for r in scene_graph.relationships])) if scene_graph.relationships else 0.90
    df_conf = float(1.0 - np.mean(depth_field.uncertainty_map))
    occ_conf = float(np.mean([o.confidence for o in occlusion_rels])) if occlusion_rels else 0.95
    overall_conf = float(0.35 * rel_conf + 0.35 * df_conf + 0.30 * occ_conf)

    spatial_confidence = SpatialConfidence(
        relationship_confidence=round(rel_conf, 4),
        depth_field_confidence=round(df_conf, 4),
        occlusion_confidence=round(occ_conf, 4),
        overall_spatial_confidence=round(overall_conf, 4)
    )

    # 7. Layer disparity metrics
    primary_ent = [e for e in entities if e.is_primary_subject][0]
    layer_disparities = compute_layer_disparity(
        camera_model, tx=0.05,
        depth_fg=primary_ent.depth_mean * 0.8,
        depth_sub=primary_ent.depth_mean,
        depth_bg=float(background_depth.max())
    )

    diagnostics = SpatialDiagnostics(
        scene_graph=scene_graph,
        depth_field=depth_field,
        occlusion_relationships=occlusion_rels,
        camera_model=camera_model,
        spatial_confidence=spatial_confidence,
        metrics_summary={
            "entity_count": len(entities),
            "relationship_count": len(scene_graph.relationships),
            "occlusion_count": len(occlusion_rels),
            "layer_disparities": layer_disparities
        }
    )

    if hash_dir is not None:
        export_spatial_diagnostics_artifacts(hash_dir, rgb_array, diagnostics)

    return diagnostics
