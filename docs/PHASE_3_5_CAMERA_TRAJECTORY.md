# Phase 3.5 Camera Trajectory Specification

## Trajectory Calibration
- Quintic smoothstep $s(t) = 6t^5 - 15t^4 + 10t^3$ for C1 velocity continuity.
- Push-In camera $T_z = -0.25$, $T_x = 0.08$, $T_y = -0.04$.
- Dolly In $T_z = -0.20$, Dolly Out $T_z = 0.20$.
- Monotonic amplitude scaling: `LOW` (0.5x) < `MEDIUM` (1.0x) < `HIGH` (2.0x).
