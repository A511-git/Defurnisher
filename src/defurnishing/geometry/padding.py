"""
Circular horizontal wrap-padding for equirectangular 360° panoramas.
Provides boundary context to diffusion inpainting to ensure seamless wrap-around seams.
"""

import numpy as np


def compute_pad_width(width: int, pad_ratio: float = 0.20) -> int:
    """Computes the integer pixel padding per side from pad_ratio."""
    return max(0, int(np.floor(width * pad_ratio)))


def wrap_pad_equirectangular(array: np.ndarray, pad_width: int) -> np.ndarray:
    """
    Circularly pads the left and right borders of a 2D or 3D equirectangular array.
    Left padding is taken from the right edge; right padding is taken from the left edge.

    Args:
        array: Array of shape (H, W) or (H, W, C).
        pad_width: Number of pixels to circularly pad on each side.

    Returns:
        Padded array of shape (H, W + 2*pad_width, [C]).
    """
    if pad_width <= 0:
        return array.copy()

    left_wrap = array[:, -pad_width:]
    right_wrap = array[:, :pad_width]
    return np.concatenate([left_wrap, array, right_wrap], axis=1)


def unpad_equirectangular(array: np.ndarray, pad_width: int, original_width: int) -> np.ndarray:
    """
    Removes the circular wrap padding, recovering the central (H, original_width, [C]) region.

    Args:
        array: Padded array of shape (H, W + 2*pad_width, [C]).
        pad_width: Padding per side originally applied.
        original_width: Original image width before padding.

    Returns:
        Unpadded array of shape (H, original_width, [C]).
    """
    if pad_width <= 0:
        return array.copy()
    return array[:, pad_width : pad_width + original_width]
