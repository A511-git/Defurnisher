"""
Alpha feathering along mask boundaries to create seamless transitions.
"""

import cv2
import numpy as np


def create_feathered_mask(
    binary_mask: np.ndarray,
    feather_radius: int = 6,
) -> np.ndarray:
    """
    Feathers the boundary of a binary mask using Gaussian blur.

    Args:
        binary_mask: Binary mask uint8 {0, 255} of shape (H, W).
        feather_radius: Pixel radius for blur smoothing.

    Returns:
        Float32 alpha mask of shape (H, W, 1) with values in range [0.0, 1.0].
        1.0 represents full inpainting, 0.0 represents original camera image.
    """
    if feather_radius <= 0:
        alpha = (binary_mask > 127).astype(np.float32)
        return alpha[..., np.newaxis]

    kernel_size = 2 * feather_radius + 1
    # Gaussian blur creates smooth linear/sigmoidal ramp across the boundary
    blurred = cv2.GaussianBlur(
        binary_mask.astype(np.float32),
        (kernel_size, kernel_size),
        sigmaX=feather_radius / 2.0,
    )
    alpha = np.clip(blurred / 255.0, 0.0, 1.0)
    return alpha[..., np.newaxis]
