import torch
from typing import Tuple, Optional
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from inference.device import get_device

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
SAM2_MODEL_ID = "facebook/sam2-hiera-tiny"
SAM2_CKPT_FILENAME = "sam2_hiera_tiny.pt"
SAM2_CONFIG_NAME = "sam2_hiera_t.yaml"

def load_depth_anything_v2(device: Optional[str] = None) -> Tuple[AutoImageProcessor, AutoModelForDepthEstimation]:
    """Loads Depth Anything V2 Small model and processor from Hugging Face."""
    if device is None:
        device = get_device()

    try:
        processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
        model.to(device)
        model.eval()
        return processor, model
    except Exception as e:
        raise RuntimeError(f"Failed to load Depth Anything V2 model from HF '{DEPTH_MODEL_ID}': {e}") from e

def load_sam2(device: Optional[str] = None) -> SAM2ImagePredictor:
    """Loads SAM 2 Hiera-Tiny model and predictor from Hugging Face."""
    if device is None:
        device = get_device()

    try:
        ckpt_path = hf_hub_download(repo_id=SAM2_MODEL_ID, filename=SAM2_CKPT_FILENAME)
        sam2_model = build_sam2(SAM2_CONFIG_NAME, ckpt_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        return predictor
    except Exception as e:
        raise RuntimeError(f"Failed to load SAM 2 model from HF '{SAM2_MODEL_ID}': {e}") from e
