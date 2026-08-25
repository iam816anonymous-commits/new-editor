from typing import Dict, Any
import torch

def profile_hardware() -> Dict[str, Any]:
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU System"
    vram_bytes = torch.cuda.get_device_properties(0).total_memory if cuda_avail else 0
    return {
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "vram_gb": round(vram_bytes / (1024**3), 2)
    }
