"""
Step 0: Local Environment Audit Script
Profiles Python runtime, PyTorch CUDA availability, CPU cores, RAM/VRAM, OpenCV, FFmpeg,
and measures actual local model loading time for Depth Anything V2 Small and SAM 2 Hiera-Tiny.
"""

import sys
import os
import platform
import time
import subprocess
import torch
import cv2
import multiprocessing
from pathlib import Path

from v0_pipeline import (
    load_depth_anything_v2,
    load_sam2,
    get_device,
    verify_ffmpeg
)

def audit_environment():
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    audit_md_path = docs_dir / "PHASE_3_LOCAL_ENVIRONMENT_AUDIT.md"

    python_ver = sys.version
    platform_system = platform.platform()

    cpu_count = multiprocessing.cpu_count()

    cuda_available = torch.cuda.is_available()
    device_name = get_device()

    vram_total_gb = 0.0
    gpu_name = "N/A (CPU Only)"
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    try:
        ffmpeg_info = verify_ffmpeg()
    except Exception as e:
        ffmpeg_info = f"FFmpeg error: {e}"

    cv2_ver = cv2.__version__
    torch_ver = torch.__version__

    print(f"[*] Auditing Local Environment...")
    print(f"    Python: {sys.version.split()[0]}")
    print(f"    PyTorch: {torch_ver} (CUDA: {cuda_available})")
    print(f"    Device: {device_name} ({gpu_name})")
    print(f"    CPU Cores: {cpu_count}")

    print("[*] Testing Depth Anything V2 Small model loading...")
    t0 = time.time()
    try:
        depth_proc, depth_model = load_depth_anything_v2(device_name)
        depth_load_time = time.time() - t0
        depth_status = f"LOADED SUCCESS ({depth_load_time:.2f}s)"
    except Exception as e:
        depth_load_time = 0.0
        depth_status = f"LOAD FAILED ({e})"

    print("[*] Testing SAM 2 Hiera-Tiny model loading...")
    t0 = time.time()
    try:
        sam2_pred = load_sam2(device_name)
        sam2_load_time = time.time() - t0
        sam2_status = f"LOADED SUCCESS ({sam2_load_time:.2f}s)"
    except Exception as e:
        sam2_load_time = 0.0
        sam2_status = f"LOAD FAILED ({e})"

    markdown_content = f"""# Phase 3 Local Environment Audit Report

## 1. System & Hardware Specifications
- **Operating System:** {platform_system}
- **Python Version:** {python_ver.split()[0]}
- **PyTorch Version:** {torch_ver}
- **OpenCV Version:** {cv2_ver}
- **FFmpeg Status:** {ffmpeg_info}

## 2. Compute Resources & Devices
- **CPU Cores:** {cpu_count} Logical Cores
- **CUDA Available:** {cuda_available}
- **Primary Inference Device:** `{device_name}`
- **GPU Device Name:** {gpu_name}
- **Total VRAM:** {vram_total_gb:.2f} GB

## 3. Real AI Model Local Loading Verification
- **Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`):** {depth_status}
- **SAM 2 Hiera-Tiny (`facebook/sam2-hiera-tiny`):** {sam2_status}

## 4. Capability Summary
- **CPU Mode:** FIRST-CLASS SUPPORTED (Target Resolution: <= 720p).
- **GPU / CUDA Mode:** {"SUPPORTED (" + gpu_name + ")" if cuda_available else "UNAVAILABLE (System operates strictly under CPU Mode with <= 720p ceiling)"}.
"""

    with open(audit_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"[✓] Environment Audit Complete! Document written to {audit_md_path}")

if __name__ == "__main__":
    audit_environment()
