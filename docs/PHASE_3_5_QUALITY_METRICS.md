# Phase 3.5 Quality Metrics Specification

## Decoupled Dual Quality Gates
1. **Motion Stability**:
   - Rigid Residual Mean: <= 0.05 px
   - Subject Edge Stability: >= 0.85
   - Subject Texture Stability: >= 0.85
   - Temporal Stability Score: >= 75.0 / 100

2. **Motion Effectiveness**:
   - Subject Scale Growth (Push-In/Dolly): 3.0% - 12.0%
   - Motion Visibility Class: `CINEMATIC`
   - Expected vs Observed Flow Agreement Score: >= 0.70
