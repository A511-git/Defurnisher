"""
Stage 4: Mask boundary feathering and high-resolution alpha compositing.
"""

from .feather import create_feathered_mask
from .composite import composite_panorama, save_final_panorama

__all__ = [
    "create_feathered_mask",
    "composite_panorama",
    "save_final_panorama",
]
