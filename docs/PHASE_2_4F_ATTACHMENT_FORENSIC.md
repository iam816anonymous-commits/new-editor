# Phase 2.4F — Attachment Graph & Thin-Structure Forensic Audit

## 1. Executive Summary
This report audits the attachment relationships in `subject_attachment_graph.json` and measures thin-structure preservation across candidate subject masks.

---

## 2. Attachment Graph Forensic Trace

| Source Candidate | Target Candidate | Inferred Attachment Type | Confidence | Supporting Evidence | Status |
|---|---|---|---|---|---|
| **Candidate #01 (Torso/Body)** | **Candidate #02 (Head/Crown)** | `RIGIDLY_ATTACHED` | `0.950` | Direct boundary contact + Depth $\Delta Z = 0.20$ | **ACCEPTED** |
| **Candidate #01 (Torso/Body)** | **Candidate #03 (Secondary Limb)** | `SOFTLY_ATTACHED` | `0.880` | Boundary contact + Depth $\Delta Z = 0.45$ | **ACCEPTED** |
| **Candidate #01 (Torso/Body)** | **Candidate #05 (Garment Extension)** | `SAME_SURFACE` | `0.850` | Boundary contact + Contour continuation | **ACCEPTED** |
| **Candidate #01 (Torso/Body)** | **Candidate #12 (Background Planet)** | `INDEPENDENT` | `0.920` | Depth $\Delta Z = 4.50 > 1.0$ threshold | **REJECTED** |

---

## 3. Thin-Structure Preservation Analysis

* **Thin Structure Candidate Area:** `1,280` pixels.
* **Preserved Area:** `1,257` pixels.
* **Lost Area:** `23` pixels ($1.8\%$).
* **Thin-Structure Preservation Ratio:** `98.2%`.
* **RGB Canny Boundary Protection:** Prevents erosion of fingers, crown points, ornaments, and garment edges during mask refinement (`refine_subject_mask`).
