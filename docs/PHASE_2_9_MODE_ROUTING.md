# Phase 2.9 — Mode Routing & Decision Matrix Report

## 1. Executive Summary
This document defines the CLI mode selection contract (`--render-mode`) and automatic routing rules for selecting between Mode A (2.5D Cinematic) and Mode B (Inferred 3D Scene).

---

## 2. CLI `--render-mode` Options

| CLI Flag | Behavior Rationale |
|---|---|
| `--render-mode 2.5d` | Explicitly forces **MODE A (2.5D Cinematic Renderer)**. Used for standard cinematic moves ($\le 15^\circ$). |
| `--render-mode 3d` | Explicitly forces **MODE B (Inferred 3D Scene Renderer)**. Used for free-viewpoint exploration ($\ge 30^\circ$) and 3D export. |
| `--render-mode auto` | Automatic router selects **EXACTLY ONE MODE** (Mode A or Mode B) based on scene complexity, camera orbit angle, hardware, and confidence. |

---

## 3. Automatic Mode Selection Matrix (`--render-mode auto`)

```
Input Image + Camera Motion Request
                 ↓
      [SceneComplexityAnalyzer]
                 ↓
      [HardwareProfile Detect]
                 ↓
     [Confidence & Orbit Check]
                 ↓
 ┌────────────────────────────────────────────────────────┐
 │                                                        │
 ↓                                                        ↓
Simple/Moderate Scene OR Orbit <= 15°       Complex Scene + GPU + Orbit >= 30°
       ↓                                                  ↓
Mode A: 2.5D Cinematic                       Mode B: Inferred 3D Scene
(quality_decision.json)                      (quality_decision.json)
```

### Routing Rules:
1. **No Hybrid Rendering:** The router selects strictly Mode A OR Mode B.
2. **Explicit User Overrides:** `--render-mode 2.5d` or `--render-mode 3d` overrides automatic selection while enforcing hard CPU resolution safety limits ($\le 720p$).
3. **Decision Logging:** Every render exports `quality_decision.json` recording `selected_mode`, `hardware`, `scene_complexity`, `source_resolution`, `reconstruction_resolution`, and `output_resolution`.
