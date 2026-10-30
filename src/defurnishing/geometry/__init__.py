"""
Equirectangular 360° and Cubemap geometry transformations.
"""

from .roll import compute_mask_centroid, roll_equirectangular, unroll_equirectangular
from .padding import wrap_pad_equirectangular, unpad_equirectangular
from .resize import resize_image_and_mask, upsample_to_size
from .transform import GeometryTransformResult, forward_geometry, inverse_geometry
from .cubemap import (
    CUBE_FACE_KEYS,
    CubemapTransformResult,
    equirectangular_to_cubemap,
    cubemap_to_equirectangular,
    get_active_faces,
    build_e2c_maps,
    build_c2e_maps,
)

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
    "CUBE_FACE_KEYS",
    "CubemapTransformResult",
    "equirectangular_to_cubemap",
    "cubemap_to_equirectangular",
    "get_active_faces",
    "build_e2c_maps",
    "build_c2e_maps",
]
