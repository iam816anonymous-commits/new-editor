# Phase 2.4D — Geometric Ground Truth Benchmark Suite

## 1. Executive Summary
This document summarizes the deterministic synthetic ground-truth benchmark suite evaluating 3D pinhole reprojection accuracy across six canonical camera trajectory styles.

## 2. Benchmark Trajectory Suite ($W=640, H=480, f_x=640.0$)

| Trajectory Style | Camera Vector $t$ / Rotation $R$ | Mean Residual | Median Residual | P95 Residual | P99 Residual | Max Residual | Status |
|---|---|---|---|---|---|---|---|
| **Pure $T_x$ Translation** | $t_x = 0.20, t_y=0, t_z=0$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | **PASS** |
| **Pure $T_y$ Translation** | $t_x = 0, t_y = -0.15, t_z=0$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | **PASS** |
| **Pure $T_z$ Push-In** | $t_x = 0, t_y = 0, t_z = -0.35$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | **PASS** |
| **Yaw Rotation** | $\text{yaw} = 1.0^\circ, t=0$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | **PASS** |
| **Pitch Rotation** | $\text{pitch} = -0.8^\circ, t=0$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | **PASS** |
| **Orbit Trajectory** | $t_x=0.08, t_y=-0.04, \text{yaw}=1.2^\circ$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | $0.00\text{px}$ | **PASS** |

## 3. Verification
All six ground-truth trajectory styles pass with zero residual flow error ($e_{\text{residual}} = 0.00\text{px}$).
