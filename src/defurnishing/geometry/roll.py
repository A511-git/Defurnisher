"""
Circular horizontal roll transformation to center the primary furniture cluster.
Centering avoids seams or cut-off artifacts at the 0°/360° boundary.
"""

from typing import Optional
import numpy as np


def compute_mask_centroid(mask: np.ndarray) -> Optional[float]:
    """
    Computes the horizontal (x-axis) centroid of the binary furniture mask.

    Args:
        mask: 2D numpy array where furniture pixels > 0.

    Returns:
        Horizontal centroid coordinate in pixels, or None if mask is empty.
    """
    y_indices, x_indices = np.nonzero(mask)
    if len(x_indices) == 0:
        return None
    return float(np.mean(x_indices))


def calculate_roll_shift(mask: np.ndarray, width: int) -> int:
    """
    Calculates the integer pixel shift Δx needed to center the mask centroid at W/2.

    Args:
        mask: 2D numpy array.
        width: Image width in pixels.

    Returns:
        Integer pixel shift Δx (positive rolls right, negative rolls left).
    """
    centroid_x = compute_mask_centroid(mask)
    if centroid_x is None:
        return 0  # No furniture detected, no shift needed
    return int(np.floor(width / 2.0 - centroid_x))


def roll_equirectangular(array: np.ndarray, shift: int) -> np.ndarray:
    """
    Circularly rolls a 2D or 3D image or mask along axis 1 (width).

    Args:
        array: Image (H, W, C) or Mask (H, W).
        shift: Number of pixels to shift circularly.

    Returns:
        Rolled numpy array of identical shape and dtype.
    """
    if shift == 0:
        return array.copy()
    return np.roll(array, shift=shift, axis=1)


def unroll_equirectangular(array: np.ndarray, shift: int) -> np.ndarray:
    """
    Reverses the circular roll by shifting by -shift along axis 1.

    Args:
        array: Image or mask array.
        shift: Original shift Δx used during forward roll.

    Returns:
        Unrolled array in the original 360° frame.
    """
    if shift == 0:
        return array.copy()
    return np.roll(array, shift=-shift, axis=1)
