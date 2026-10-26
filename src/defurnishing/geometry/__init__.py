"""
Equirectangular 360° geometry transformations: roll, wrap-padding, resizing, and invertible pipelines.
"""

from .roll import compute_mask_centroid, roll_equirectangular, unroll_equirectangular
from .padding import wrap_pad_equirectangular, unpad_equirectangular
from .resize import resize_image_and_mask, upsample_to_size
from .transform import GeometryTransformResult, forward_geometry, inverse_geometry

__all__ = [
    "compute_mask_centroid",
    "roll_equirectangular",
    "unroll_equirectangular",
    "wrap_pad_equirectangular",
    "unpad_equirectangular",
    "resize_image_and_mask",
    "upsample_to_size",
    "GeometryTransformResult",
    "forward_geometry",
    "inverse_geometry",
]
