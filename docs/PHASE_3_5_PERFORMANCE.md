# Phase 3.5 Performance Profile

## Runtime Profile (320x320 Input, 48 Frames)
- **Depth Inference (Depth Anything V2 Small)**: ~0.9s
- **Segmentation (SAM 2 Hiera-Tiny)**: ~5.5s
- **Scene Analysis & Background Inpainting**: ~2.5s
- **Frame Sequence View Synthesis (48 frames)**: ~22s (~0.45s / frame)
- **FFmpeg MP4 Encoding & Verification**: ~1.2s
- **Total Pipeline Runtime**: ~32s - 34s
