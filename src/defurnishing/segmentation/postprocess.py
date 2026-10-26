"""
Post-processing operations for furniture segmentation masks:
morphological dilation to swallow contact shadows and small noise removal.
"""

import cv2
import numpy as np


def dilate_mask(mask: np.ndarray, dilate_pixels: int = 10) -> np.ndarray:
    """
    Applies morphological dilation with an elliptical kernel.
    Dilating the furniture mask expands it over floor contact shadows,
    ensuring they are inpainted cleanly rather than left behind as dark smudges.

    Args:
        mask: Binary mask uint8 {0, 255}.
        dilate_pixels: Pixel radius for morphological dilation.

    Returns:
        Dilated binary mask uint8 {0, 255}.
    """
    if dilate_pixels <= 0:
        return mask.copy()

    kernel_size = 2 * dilate_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated


def clean_mask(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """
    Removes very small isolated pixel clusters below min_area.

    Args:
        mask: Binary mask uint8 {0, 255}.
        min_area: Minimum connected component area in pixels.

    Returns:
        Cleaned binary mask.
    """
    if min_area <= 0:
        return mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned
