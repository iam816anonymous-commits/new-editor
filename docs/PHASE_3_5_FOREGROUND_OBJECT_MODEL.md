# Phase 3.5 Foreground Object Model Specification

## Foreground Parallax
- Foreground depth layer receives 2.20x disparity multiplier relative to background.
- Preserves soft distance-transform feathering and rigid interior core.
- Enforces depth-weighted layer motion ordering (`FOREGROUND` > `MIDGROUND` > `BACKGROUND`).
