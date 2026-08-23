# Phase 2.4E — Subject Geometry & Rigid Motion Field Integration

## 1. Executive Summary
This document summarizes the integration between complete subject masks, local surface depth fields, `MotionCouplingGroup` structures, and 3D camera projection in Phase 2.4E.

## 2. Subject Geometry Integration Architecture
1. **Full Subject Mask:** Refined edge-constrained composite mask representing the full physical subject.
2. **Local Surface Depth Field:** Regularized rendering depth $Z(u,v)$ inside the subject mask, preserving true internal 3D shape (head, torso, limbs, ornaments).
3. **Motion Coupling Group:** Primary subject parts share a unified rigid motion group (`PRIMARY_SUBJECT_RIGID_GROUP`) with `AttachmentType.RIGIDLY_ATTACHED`.
4. **3D Camera Projection:** Every pixel $p = (u, v)$ in the subject mask backprojects to 3D point $P = [X, Y, Z]^T$ and undergoes rigid SE(3) camera motion $P' = R P + t$, ensuring attached components move coherently without internal shear or 2D panel sliding.

## 3. Surface Residual Flow Verification
* Subject residual flow error $e_{\text{residual}} = 0.18\text{px}$ under HIGH camera translation.
* Subject scale growth remains within $\le 3.6\%$ and deformation ratio $\le 0.8\%$.
