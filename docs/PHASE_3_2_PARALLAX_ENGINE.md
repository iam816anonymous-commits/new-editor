# Phase 3.2 Parallax Engine Architecture

## 1. Parallax Mathematical Formulation
The Mode A 2.5D view synthesis engine models differential camera motion using perspective back-projection and subpixel forward splatting.

Given pixel $(u, v)$ with normalized rendering coordinate depth $Z(u, v) \in [0.1, 10.0]$:

$$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}$$

Under camera rigid body transformation $(R, t)$:

$$P' = P \cdot R^T + t \cdot m_{\text{layer}}$$

Projected 2D coordinates:

$$u' = f_x \cdot \frac{X'}{Z'} + c_x, \quad v' = f_y \cdot \frac{Y'}{Z'} + c_y$$

Image-space displacement $\Delta u \propto \frac{1}{Z}$.

## 2. Layer Motion Multipliers
The layer motion multiplier map $m_{\text{layer}}$ enforces strict depth-weighted layer ordering:
- `FOREGROUND`: $2.20\times$ multiplier
- `MIDGROUND`: $1.50\times$ multiplier
- `BACKGROUND`: $1.00\times$ multiplier
- `PRIMARY_SUBJECT`: $0.35\times$ multiplier (subject stability anchor)
