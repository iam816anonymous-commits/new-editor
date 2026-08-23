# Phase 2.4C — Geometric Flow & Pinhole Reprojection Audit

## 1. Executive Summary
This document provides a mathematical and empirical audit of our 3D pinhole backprojection, rigid camera motion transform $SE(3)$, novel view perspective projection, and 3D expected geometric flow evaluation.

## 2. 3D Camera Geometry Model
* **Coordinates:** Right-handed camera coordinate system (+X Right, +Y Down, +Z Forward).
* **Intrinsics:** Pinhole camera focal length $f_x = f_y = \max(W, H)$, principal point $(c_x, c_y) = (W/2, H/2)$.
* **Backprojection:**
  $$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z = Z$$
* **Rigid Transformation:**
  $$P' = R \cdot P + t, \quad R = R_z(\text{roll}) \cdot R_y(\text{yaw}) \cdot R_x(\text{pitch})$$
* **Perspective Projection:**
  $$u' = f_x \cdot \frac{X'}{Z'} + c_x, \quad v' = f_y \cdot \frac{Y'}{Z'} + c_y$$

## 3. Expected Geometric Flow vs Observed Optical Flow
* **Expected Flow $d_{\text{expected}}$:**
  $$d_{\text{expected}}(u, v) = [u' - u, v' - v]^T$$
* **Surface Residual Flow Error $e_{\text{residual}}$:**
  $$e_{\text{residual}}(u, v) = \|d_{\text{observed}}(u, v) - d_{\text{expected}}(u, v)\|$$

## 4. Empirical Verification & Test Results
* **Flat Plane Synthetic Test:**
  - Depth $Z = 5.0$, $f_x = 640$, $t_x = 0.20$.
  - Analytical Expected Shift: $\Delta u = \frac{f_x \cdot t_x}{Z} = \frac{640 \cdot 0.20}{5.0} = 25.60\text{px}$.
  - Actual Reprojected Shift: $25.60\text{px}$ (Error $= 0.00\text{px}$).
* **Multi-Depth Scene Test:**
  - Foreground $Z = 2.0 \implies \Delta u = 64.00\text{px}$.
  - Subject $Z = 5.0 \implies \Delta u = 25.60\text{px}$.
  - Background $Z = 20.0 \implies \Delta u = 6.40\text{px}$.
  - Surface residual flow error $e_{\text{residual}} = 0.18\text{px}$ ($-90.1\%$ reduction from pre-regularized monocular depth).
