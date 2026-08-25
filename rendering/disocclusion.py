import cv2
import numpy as np

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
