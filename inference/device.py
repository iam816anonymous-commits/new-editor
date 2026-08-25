import torch
from typing import Optional

def get_device() -> str:
    """Selects CUDA if available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"
