# Phase 3.4 Subject Integrity Specification

## Overview
Subject Integrity evaluates structural preservation and temporal lock quality across rendered subject keyframes.

## Metrics
- `residual_rigid_mean`: Mean residual displacement error after rigid SE(2) alignment.
- `subject_edge_stability`: Boundary displacement stability across rendered keyframes.
- `subject_texture_stability`: Pixel-level structural similarity inside subject mask.
- `subject_integrity_score`: Composite score (0-100 scale), required to be >= 75.0.
