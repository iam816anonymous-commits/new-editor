# Phase 3.2 Cinematic Motion & Strength Tiers

## 1. Trajectory Easing & Continuity
Camera trajectories use quintic smoothstep easing:

$$s(t) = 6t^5 - 15t^4 + 10t^3 \quad (t \in [0, 1])$$

Satisfying $s(0)=0, s(1)=1$, and $s'(0) = s'(1) = 0$ ($C^1$ velocity continuity without start/stop jerk).

## 2. Motion Strength Tiers
- **`subtle`**: Conservative displacement ceiling ($3.0\%$ of max image dimension). Minimal disocclusion exposure (~92.6 KB video size).
- **`cinematic` / `balanced`**: Standard production displacement ceiling ($6.0\%$ of max image dimension). Noticeable depth parallax (~121.9 KB video size).
- **`strong`**: High-impact stress test displacement ceiling ($10.0\%$ of max image dimension). Maximum safe camera travel.
