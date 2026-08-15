# First-Principles Cinematic 2.5D Parallax Renderer (V0)

A standalone, deterministic Python rendering engine that transforms a single 2D RGB image into a cinematic camera-motion video with realistic depth and 3D parallax.

---

## 1. Project Architecture

The pipeline uses real AI models and classic 3D pinhole camera reprojection with forward subpixel splatting and deterministic Z-buffering:

```
USER IMAGE
    ↓
IMAGE VALIDATION & SHA-256 HASH
    ↓
REAL DEPTH ANYTHING V2 SMALL
    ↓
DEPTH NORMALIZATION & REFINEMENT
    ↓
REAL SAM 2 HIERA-TINY
    ↓
SUBJECT MASK & GEOMETRY
    ↓
BACKGROUND RGB & DEPTH INPAINTING
    ↓
AUTOMATIC MOTION PLANNING & CLOSED-LOOP SAFETY ENVELOPE
    ↓
3D REPROJECTION & FORWARD SPLATTING (Z-BUFFER)
    ↓
TEMPORALLY STABLE FRAME GENERATION
    ↓
MP4 ENCODING (FFMPEG)
```

---

## 2. Models Used & Commercial Licensing

| Task | Model / Checkpoint | License | Source / Link |
| :--- | :--- | :--- | :--- |
| **Monocular Depth Estimation** | `depth-anything/Depth-Anything-V2-Small-hf` | Apache-2.0 | [Hugging Face](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| **Subject Segmentation** | `facebook/sam2-hiera-tiny` | Apache-2.0 | [Hugging Face](https://huggingface.co/facebook/sam2-hiera-tiny) |

*All selected models use permissive open-source licenses suitable for commercial use.*

---

## 3. Installation & Setup

### Prerequisites
- Python 3.11+
- FFmpeg installed and available on system `PATH`
- PyTorch with CUDA support (optional, falls back automatically to CPU)

### Installation
```bash
pip install -r requirements.txt
```

---

## 4. CLI Usage

Run the pipeline on any user-provided image:

```bash
python v0_pipeline.py --input "path/to/image.jpg" --motion Orbit --strength Cinematic
```

### Options
- `--input <PATH>`: Path to the input image file (Required).
- `--motion <TYPE>`: Trajectory pattern (`Dolly In`, `Dolly Out`, `Horizontal Pan`, `Vertical Pan`, `Orbit`, `Micro Orbit`). Default: `Orbit`.
- `--strength <LEVEL>`: Motion strength limit (`Subtle`, `Cinematic`, `Strong`). Default: `Cinematic`.
- `--output-dir <PATH>`: Base output directory (Default: `output`).

---

## 5. Output Directory Structure

For an input image, the pipeline computes its SHA-256 hash and creates:

```
output/<short_hash>/
├── original.png
├── depth.png
├── subject_mask.png
├── background_plate.png
├── background_depth.png
├── confidence_map.png
├── subtle/
│   ├── frame_00.png
│   ├── frame_mid.png
│   ├── frame_last.png
│   ├── output.mp4
│   └── metrics.json
├── cinematic/
└── strong/
```

---

## 6. Mathematical Model

### Pinhole Camera Back-Projection
For pixel $(u, v)$ with estimated depth $Z$:

$$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z = Z$$

### 3D Transformation & Projection
For camera translation $t$ and rotation matrix $R$:

$$P' = R \cdot [X, Y, Z]^T + t$$

$$u' = f_x \cdot \frac{X'}{Z'} + c_x, \quad v' = f_y \cdot \frac{Y'}{Z'} + c_y$$

---

## 7. Development & Testing

Run unit and integration tests:

```bash
python -m pytest
```

Level 1 tests verify pure 3D camera geometry math, trajectory generation, and CLI setup. Level 2 tests verify real model loading and tensor outputs. Level 3 tests verify full E2E pipeline execution.
