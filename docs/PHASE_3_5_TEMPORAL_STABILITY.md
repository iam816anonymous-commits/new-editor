# Phase 3.5 Temporal Stability Specification

## Temporal Smoothness
- Smooth C1 quintic smoothstep acceleration/deceleration for non-looping trajectories ($P(0) \neq P(1), V(0) = V(1) = 0$).
- Windowed cosine loop closure error $P(0) == P(1), V(0) == V(1) == 0$ for cyclic Orbit trajectories.
- Motion-aware temporal stability score evaluates frame-to-frame MAD without dampening camera-projected travel.
