"""
Unified pipeline transformations: forward preparation and inverse coordinate restoration.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..config.schema import GeometryConfig
from .padding import compute_pad_width, unpad_equirectangular, wrap_pad_equirectangular
from .resize import compute_inpaint_dimensions, resize_image_and_mask, upsample_to_size
from .roll import calculate_roll_shift, roll_equirectangular, unroll_equirectangular


@dataclass
class GeometryTransformResult:
    """Metadata and transformed arrays from forward geometric pre-processing."""
    transformed_image: np.ndarray
    transformed_mask: np.ndarray
    orig_width: int
    orig_height: int
    shift_x: int
    pad_width: int
    padded_width: int
    padded_height: int
    inpaint_width: int
    inpaint_height: int

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary of metadata."""
        return {
            "orig_width": self.orig_width,
            "orig_height": self.orig_height,
            "shift_x": self.shift_x,
            "pad_width": self.pad_width,
            "padded_width": self.padded_width,
            "padded_height": self.padded_height,
            "inpaint_width": self.inpaint_width,
            "inpaint_height": self.inpaint_height,
        }

    def save_meta(self, output_path: Path) -> None:
        """Saves metadata dictionary as JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_meta(cls, meta_path: Path) -> Dict[str, Any]:
        """Loads metadata dictionary from JSON file."""
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)


def forward_geometry(
    image: np.ndarray,
    mask: np.ndarray,
    config: GeometryConfig,
) -> GeometryTransformResult:
    """
    Executes the forward geometric transformation:
    1. Circular Roll: Centering the mask along horizontal axis (if config.roll).
    2. Circular Wrap-Pad: Appending boundary context margins (if config.pad).
    3. Resizing: Scaling image & mask to diffusion working resolution (divisible by 8).

    Args:
        image: Original RGB image (H, W, 3).
        mask: Binary mask (H, W).
        config: GeometryConfig instance.

    Returns:
        GeometryTransformResult containing transformed arrays and restoration metadata.
    """
    orig_h, orig_w = image.shape[:2]

    # 1. Roll (Centering)
    shift_x = 0
    if config.roll:
        shift_x = calculate_roll_shift(mask, orig_w)
        image = roll_equirectangular(image, shift_x)
        mask = roll_equirectangular(mask, shift_x)

    # 2. Wrap-Padding
    pad_width = 0
    if config.pad:
        pad_width = compute_pad_width(orig_w, config.pad_ratio)
        image = wrap_pad_equirectangular(image, pad_width)
        mask = wrap_pad_equirectangular(mask, pad_width)

    padded_h, padded_w = image.shape[:2]

    # 3. Downsampling / Resizing
    inpaint_w, inpaint_h = compute_inpaint_dimensions(padded_w, padded_h, config.target_height)
    inpaint_image, inpaint_mask = resize_image_and_mask(image, mask, inpaint_w, inpaint_h)

    return GeometryTransformResult(
        transformed_image=inpaint_image,
        transformed_mask=inpaint_mask,
        orig_width=orig_w,
        orig_height=orig_h,
        shift_x=shift_x,
        pad_width=pad_width,
        padded_width=padded_w,
        padded_height=padded_h,
        inpaint_width=inpaint_w,
        inpaint_height=inpaint_h,
    )


def inverse_geometry(
    inpaint_raw: np.ndarray,
    meta: Dict[str, Any],
) -> np.ndarray:
    """
    Reverses the forward geometric transformation:
    1. Upsample raw inpainting back to high-resolution padded dimensions.
    2. Unpad: Slice out wrap padding to restore original image width.
    3. Unroll: Reverse circular roll by -shift_x to restore original 360° orientation.

    Args:
        inpaint_raw: Generated inpainting result (H_inpaint, W_inpaint, 3).
        meta: Metadata dictionary produced by GeometryTransformResult.

    Returns:
        Restored image array of original dimensions (orig_height, orig_width, 3).
    """
    orig_w = meta["orig_width"]
    orig_h = meta["orig_height"]
    shift_x = meta["shift_x"]
    pad_width = meta["pad_width"]
    padded_w = meta["padded_width"]

    # 1. Upsample back to padded dimensions
    upscaled = upsample_to_size(inpaint_raw, target_w=padded_w, target_h=orig_h)

    # 2. Unpad
    unpadded = unpad_equirectangular(upscaled, pad_width=pad_width, original_width=orig_w)

    # 3. Unroll
    restored = unroll_equirectangular(unpadded, shift=shift_x)

    return restored
