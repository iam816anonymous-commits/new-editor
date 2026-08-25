import cv2
import numpy as np
from typing import Tuple

def reconstruct_background_rgb(
    rgb_array: np.ndarray,
    dilated_mask: np.ndarray,
    inpaint_radius: int = 5
) -> np.ndarray:
    """
    Creates a clean background plate from the original reference image using conservative inpainting.
    Strictly preserves observed background pixels outside dilated_mask.
    """
    mask_uint8 = (dilated_mask * 255).astype(np.uint8)

    # Inpaint missing subject region using Navier-Stokes (INPAINT_NS) / Telea algorithm
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(bgr_array, mask_uint8, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    # Strictly enforce observed pixel preservation: non-mask pixels are copied verbatim from original
    bg_plate = rgb_array.copy()
    bg_plate[dilated_mask] = inpainted_rgb[dilated_mask]

    return bg_plate

def complete_background_depth(
    depth_map: np.ndarray,
    dilated_mask: np.ndarray,
    inpaint_radius: int = 7
) -> np.ndarray:
    """
    Completes background depth map separately from RGB reconstruction.
    Propagates surrounding background depth into the subject area using smooth boundary extrapolation,
    strictly preserving known observed background depth pixels outside dilated_mask.
    """
    d_min, d_max = depth_map.min(), depth_map.max()
    depth_span = max(d_max - d_min, 1e-5)

    depth_norm = ((depth_map - d_min) / depth_span * 255.0).astype(np.uint8)
    mask_uint8 = (dilated_mask * 255).astype(np.uint8)

    # Inpaint depth using Navier-Stokes boundary propagation
    inpainted_norm = cv2.inpaint(depth_norm, mask_uint8, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)
    inpainted_depth = d_min + (inpainted_norm.astype(np.float32) / 255.0) * depth_span

    # Strictly enforce observed background depth preservation
    bg_depth = depth_map.copy()
    bg_depth[dilated_mask] = inpainted_depth[dilated_mask]

    return bg_depth.astype(np.float32)

def compute_provenance_map(dilated_mask: np.ndarray) -> np.ndarray:
    """
    Computes pixel provenance map:
    1.0 = OBSERVED (original reference pixel)
    0.0 = RECONSTRUCTED (inpainted pixel)
    """
    provenance = np.ones(dilated_mask.shape, dtype=np.float32)
    provenance[dilated_mask] = 0.0
    return provenance


def inpaint_hidden_regions(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
    radius: int = 5
) -> np.ndarray:
    """
    Pre-animation inpainting of hidden/disoccluded regions prior to camera movement.
    Supports 'telea' (Fast Marching) and 'ns' (Navier-Stokes).
    Handles both 3-channel RGB images and single-channel depth/gray arrays.
    """
    mask_uint8 = (mask.astype(bool) * 255).astype(np.uint8)
    inpaint_flag = cv2.INPAINT_NS if method.lower() == "ns" else cv2.INPAINT_TELEA

    if image.ndim == 3 and image.shape[2] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(bgr, mask_uint8, inpaintRadius=radius, flags=inpaint_flag)
        return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Single channel (depth map or grayscale)
        d_min, d_max = float(image.min()), float(image.max())
        d_span = max(d_max - d_min, 1e-5)
        norm = ((image - d_min) / d_span * 255.0).astype(np.uint8)
        inpainted_norm = cv2.inpaint(norm, mask_uint8, inpaintRadius=radius, flags=inpaint_flag)
        return (d_min + (inpainted_norm.astype(np.float32) / 255.0) * d_span).astype(image.dtype)


def extrapolate_edge_padding(
    image: np.ndarray,
    edge_mask: np.ndarray,
    pad_size: int = 15
) -> np.ndarray:
    """
    Extrapolates texture into occlusion edge zones to prevent tearing during camera movement.
    Dilates the edge mask by pad_size and inpaints using background texture features.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad_size * 2 + 1, pad_size * 2 + 1))
    dilated_edges = cv2.dilate((edge_mask.astype(bool) * 255).astype(np.uint8), kernel) > 0
    return inpaint_hidden_regions(image, dilated_edges, method="telea", radius=pad_size)


class PersistentBackgroundCanvas:
    """
    Maintains a persistent background plate across frame rendering to eliminate
    disocclusion inpainting flicker and temporal texture instability.
    """
    def __init__(self, initial_bg_rgb: np.ndarray, initial_bg_depth: np.ndarray):
        self.bg_rgb = initial_bg_rgb.copy()
        self.bg_depth = initial_bg_depth.copy()
        self.updated_mask = np.zeros(initial_bg_rgb.shape[:2], dtype=bool)

    def update_canvas(self, new_rgb: np.ndarray, new_depth: np.ndarray, newly_exposed_mask: np.ndarray):
        """Updates persistent background canvas with newly observed/inpainted pixels."""
        if not np.any(newly_exposed_mask):
            return
        m = newly_exposed_mask.astype(bool)
        self.bg_rgb[m] = new_rgb[m]
        self.bg_depth[m] = new_depth[m]
        self.updated_mask[m] = True

    def get_canvas(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.bg_rgb, self.bg_depth
