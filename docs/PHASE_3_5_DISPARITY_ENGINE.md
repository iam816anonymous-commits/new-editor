# Phase 3.5 Disparity Engine Specification

## Disparity Mapping & Layer Motion
1. **Depth Normalization**: Continuous normalized rendering depth $Z \in [0.10, 10.0]$.
2. **Layer Multipliers**:
   - `FOREGROUND`: 2.20x
   - `MIDGROUND`: 1.50x
   - `BACKGROUND`: 1.00x
   - `PRIMARY_SUBJECT`: 1.0x Z translation for perspective scale growth, 0.75x XY for lateral anchor stability.
