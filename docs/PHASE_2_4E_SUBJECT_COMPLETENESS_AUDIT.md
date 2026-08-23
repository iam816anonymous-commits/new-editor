# Phase 2.4E — Subject Completeness & Selection Forensic Data Flow Audit

## 1. Executive Summary
This document records a forensic line-by-line audit of the subject selection and mask refinement pipeline (`subject_selection/` and `v0_pipeline.py`).

The objective is to identify why primary subjects were sometimes partially truncated or missing attached components (limbs, crowns, ornaments, weapons, garments) and establish the architectural enhancements required for **Full Subject Completeness**.

---

## 2. Forensic Audit Findings & Root Causes

1. **Top-K Candidate Truncation in Grouping (`candidate_grouper.py`, line 65):**
   - Candidate grouping previously evaluated pairwise combinations across only the top 6 candidates (`top_feats[:6]`).
   - Small attached components (e.g. candidate IDs 7..15 representing ornaments, hands, crown points, or secondary limbs) were discarded before compound subject graph evaluation.
2. **Hard Spatial Proximity Cutoff (`candidate_grouper.py`, line 42):**
   - Proximity evaluation used a strict 50px euclidean threshold (`spatial_proximity_threshold_px = 50.0`).
   - Disconnected or partially occluded attached parts (e.g., hand held out, or crown tip) separated by small gaps received `spatial_compat = 0.0`.
3. **Lack of Partial-Occlusion Subject Continuation Graph:**
   - When a limb or ornament disappears behind an occluding garment or prop and reappears nearby, the system previously treated the reappearing component as an independent object or background element.
4. **Aggressive Morphological Mask Erasure in Refinement (`mask_merger.py`):**
   - Mask refinement used fixed morphological opening kernel operations without edge constraints, which accidentally erased thin structures (fingers, ornament spikes, thin weapons).
5. **Absence of Edge-Constrained Mask Expansion:**
   - Candidate mask merging relied on binary mask union without tracing Canny RGB boundary edges, causing mask expansion to stop prematurely at SAM candidate boundaries rather than expanding to the true object boundary.

---

## 3. Required Architectural Fixes
* Expand candidate pool evaluation to consider all candidates in the candidate graph.
* Implement multi-signal attachment scoring (boundary proximity, RGB Canny edge continuity, contour continuation, depth continuity, texture, occlusion continuation).
* Implement edge-constrained mask expansion in `mask_merger.py` that expands candidate boundaries up to strong RGB Canny boundaries while stopping at strong object boundaries.
* Export `subject_completeness.json` and visual diagnostic packages.
