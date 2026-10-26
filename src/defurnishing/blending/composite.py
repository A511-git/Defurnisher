"""
High-resolution compositing and output generation.
Preserves 100% of untouched camera pixels (walls, ceiling, untouched floor).
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from .feather import create_feathered_mask


def composite_panorama(
    original_image: np.ndarray,
    restored_inpaint: np.ndarray,
    binary_mask: np.ndarray,
    feather_radius: int = 6,
) -> np.ndarray:
    """
    Blends the geometrically restored inpaint over the original high-resolution panorama.

    Args:
        original_image: Original camera image (H, W, 3) uint8.
        restored_inpaint: Restored inpainting (H, W, 3) uint8.
        binary_mask: Final furniture mask (H, W) uint8 {0, 255}.
        feather_radius: Boundary feathering radius in pixels.

    Returns:
        Final composite RGB image (H, W, 3) uint8.
    """
    alpha = create_feathered_mask(binary_mask, feather_radius=feather_radius)

    orig_f = original_image.astype(np.float32)
    inpaint_f = restored_inpaint.astype(np.float32)

    composite_f = (inpaint_f * alpha) + (orig_f * (1.0 - alpha))
    composite_uint8 = np.clip(composite_f, 0, 255).astype(np.uint8)

    return composite_uint8


def save_final_panorama(
    image: np.ndarray,
    output_path: Path,
    output_format: str = "jpg",
    jpeg_quality: int = 95,
) -> None:
    """
    Saves the final composite panorama to disk with high fidelity.

    Args:
        image: RGB image array (H, W, 3) uint8.
        output_path: Path to save file.
        output_format: 'jpg' or 'png'.
        jpeg_quality: Quality (1-100) for JPEG output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_format.lstrip(".").lower()

    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if fmt in ("jpg", "jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, jpeg_quality))]
        cv2.imwrite(str(output_path), bgr_image, params)
    else:
        cv2.imwrite(str(output_path), bgr_image)
