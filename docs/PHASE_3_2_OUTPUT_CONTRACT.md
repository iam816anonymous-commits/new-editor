# Phase 3.2 Output Contract Forensic Mapping

## 1. Expected Output Directory Structure & Artifact Ownership

```
output/<short_hash>/
├── 01_original.png                           # Reference input image
├── 02_depth.png                              # Depth Anything V2 depth visualization
├── 03_candidate_contact_sheet.png            # SAM 2 candidate mask contact sheet
├── 04_candidate_scores.png                   # Candidate multi-signal score breakdown
├── 05_candidate_groups.png                   # Grouped subject candidate masks
├── 06_selected_group.png                     # Selected primary subject mask
├── 07_raw_subject_mask.png                   # Raw SAM 2 mask
├── 08_refined_subject_mask.png               # Edge-refined subject mask
├── 09_subject_overlay.png                    # Subject mask RGB overlay
├── 10_depth_subject_overlay.png              # Depth map subject mask overlay
├── 11_background_contamination.png           # Background contamination risk map
├── 12_validation_report.json                 # Candidate selection validation JSON
├── original.png                              # Copy of original reference image
├── depth.png                                 # Normalized depth map
├── subject_mask.png                          # Final subject mask
├── confidence_map.png                        # Depth confidence map
├── candidate_selection.json                  # Candidate selection traceability JSON
├── background_plate.png                      # Reconstructed clean background plate
├── background_depth.png                      # Extrapolated background depth map
├── provenance_map.png                        # Pixel provenance (OBSERVED vs RECONSTRUCTED)
├── boundary_risk_map.png                     # Silhouette boundary exposure risk
├── camera_path.png                           # Planned 3D camera trajectory plot
├── camera_path.json                          # Planned 3D camera poses JSON
├── camera_vs_raster_motion.png               # Camera intent vs raster motion plot
├── layer_displacement_curves.png             # Layer displacement curves plot
├── final_contact_sheet.png                   # 5-keyframe sequence contact sheet
├── phase_1_7_visual_validation_contact_sheet.png # Multi-row visual validation contact sheet
├── motion_amplitude_comparison.png           # LOW vs MEDIUM vs HIGH amplitude comparison
├── temporal_diagnostics.png                  # Frame-to-frame temporal MAD curve plot
├── metrics.json                              # Main pipeline performance & inference metrics
├── motion_report.json                        # Detailed 3D motion & parallax quality metrics
├── validation_summary.json                   # 4-tier pass validation summary
├── spatial_scene.json                        # Spatial Intelligence scene graph JSON
├── spatial_relationships.json                # Spatial entity pairwise relationships
├── spatial_diagnostics.json                  # Spatial engine diagnostic metadata
├── subject_attachment_graph.json             # Subject attachment graph JSON
├── subject_attachment_graph.png              # Subject attachment graph visualization
├── subject_completeness.json                 # Subject completeness metrics
├── subject_temporal_stability.json           # Subject temporal stability score
├── subtle/                                    # Subtle strength render folder
├── cinematic/                                 # Cinematic strength render folder
│   ├── visual_review.png                     # 2x4 visual review contact sheet
│   ├── visual_review_diagnostics.png         # Visual review 2x enlarged crops
│   ├── output.mp4                            # Backwards-compatible video render
│   └── frames/                               # Sequence frames folder
│       ├── frame_0000.png
│       └── ...
├── strong/                                    # Strong strength render folder
├── output_video/                              # Dedicated final video folder
│   └── cinematic_<hash>_<timestamp>.mp4      # Unique timestamped video render
├── debug/                                     # Raster debug & optical flow trace folder
│   ├── depth_distribution.json
│   ├── frame_diff_f00_f99.png
│   ├── frame_difference_report.json
│   ├── frame_overlay_f00_f99.png
│   ├── motion_heatmap.png
│   ├── pixel_trajectory.png
│   ├── projected_vs_raster_trajectory.png
│   ├── raster_trace.json
│   └── temporal_motion_profile.json
└── spatial_analysis/                          # Spatial Intelligence diagnostic folder
    ├── 00_primary_spatial_overlay.png
    ├── 01_depth_raw.png
    ├── 02_depth_edges.png
    ├── 15_splat_coverage.png
    ├── 16_zbuffer_ownership.png
    ├── attachment_graph.json
    ├── disocclusion_forecast.json
    ├── motion_coupling.json
    ├── motion_eligibility.json
    ├── parallax_regions.json
    └── scene_analysis.json
```

## 2. Operation Mapping Matrix

| Operation | Original Monolith Function | Modular Location | Status |
|-----------|---------------------------|------------------|--------|
| Hash Directory Setup | `setup_output_directories` | `output/artifacts.py` | Active |
| Phase B Artifacts | `save_phase_b_diagnostic_artifacts` | `output/artifacts.py` | Active |
| Phase C Artifacts | `save_phase_c_diagnostic_artifacts` | `output/artifacts.py` | Active |
| Spatial Scene Analysis | `analyze_spatial_scene` | `spatial_intelligence/spatial_engine.py` | Active |
| Frame Sequence Rendering | `render_full_frame_sequence` | `rendering/sequence_renderer.py` | Active |
| MP4 Video Encoding | `encode_and_verify_mp4` | `rendering/video_encoder.py` | Active |
| Output Video Directory | N/A (New User Requirement) | `app/application.py` -> `output_video/` | Active |
| Output Contract Validation | N/A (New Security Requirement) | `output/artifacts.py:validate_output_contract` | Active |
