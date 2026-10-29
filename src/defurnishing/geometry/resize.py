"""
Aspect-ratio preserving resizing for diffusion inpainting models.
Ensures both width and height are strictly divisible by 8 (required by VAE latent downsamplers).
"""

from typing import Tuple
import cv2
import numpy as np


def compute_inpaint_dimensions(
    current_w: int,
    current_h: int,
    target_height: int = 256,
) -> Tuple[int, int]:
    """
    Computes scaled (width, height) preserving aspect ratio, ensuring both are multiples of 8.

    Args:
        current_w: Width of image.
        current_h: Height of image.
        target_height: Requested working height.

    Returns:
        Tuple of (new_width, new_height), both divisible by 8.
    """
    aspect = current_w / float(current_h)
    new_h = int(round(target_height / 8.0)) * 8
    new_w = int(round((new_h * aspect) / 8.0)) * 8
    return max(8, new_w), max(8, new_h)


def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_w: int,
    target_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes image using Lanczos interpolation and mask using Nearest Neighbor.

    Args:
        image: RGB image array of shape (H, W, 3).
        mask: Binary mask array of shape (H, W).
        target_w: Target width.
        target_h: Target height.

    Returns:
        Tuple of (scaled_image, scaled_mask).
    """
    # Downscaling uses AREA for smoother anti-aliasing; upscaling uses LANCZOS4
    current_h, current_w = image.shape[:2]
    interp = cv2.INTER_AREA if (target_w < current_w or target_h < current_h) else cv2.INTER_LANCZOS4

    scaled_image = cv2.resize(image, (target_w, target_h), interpolation=interp)
    scaled_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Ensure mask remains strictly binary {0, 255}
    _, scaled_mask = cv2.threshold(scaled_mask, 127, 255, cv2.THRESH_BINARY)
    return scaled_image, scaled_mask


def upsample_to_size(
    image: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """
    Upsamples an inpainted image back to high-resolution padded dimensions using Lanczos.

    Args:
        image: Inpainted image array (H_inpaint, W_inpaint, 3).
        target_w: Padded high-res width.
        target_h: Original high-res height.

    Returns:
        Upscaled image array of shape (target_h, target_w, 3).
    """
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
