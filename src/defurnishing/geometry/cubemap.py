"""
High-performance vectorized Equirectangular <-> Cubemap conversions using OpenCV remap.
Converts between 360° equirectangular panoramas and 6 rectilinear cube faces:
Front (F), Right (R), Back (B), Left (L), Up (U), Down (D).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

CUBE_FACE_KEYS = ["F", "R", "B", "L", "U", "D"]


@dataclass
class CubemapTransformResult:
    """Holds projected cube faces and metadata for the cubemap inpainting pipeline."""
    img_faces: Dict[str, np.ndarray]
    mask_faces: Dict[str, np.ndarray]
    active_faces: List[str]
    orig_width: int
    orig_height: int
    inpainted_faces: Dict[str, np.ndarray] = field(default_factory=dict)


def _build_e2c_face_rays(face: str, face_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 3D ray vectors (X, Y, Z) for each pixel of a cube face.
    Face coordinates u, v in [-1, 1].
    """
    j, i = np.meshgrid(
        np.arange(face_size, dtype=np.float32),
        np.arange(face_size, dtype=np.float32),
    )
    u = (2.0 * (j + 0.5) / face_size) - 1.0
    v = 1.0 - (2.0 * (i + 0.5) / face_size)

    if face == "F":    # Front (+Z)
        x = u
        y = v
        z = np.ones_like(u)
    elif face == "R":  # Right (+X)
        x = np.ones_like(u)
        y = v
        z = -u
    elif face == "B":  # Back (-Z)
        x = -u
        y = v
        z = -np.ones_like(u)
    elif face == "L":  # Left (-X)
        x = -np.ones_like(u)
        y = v
        z = u
    elif face == "U":  # Up (+Y)
        x = u
        y = np.ones_like(u)
        z = -v
    elif face == "D":  # Down (-Y)
        x = u
        y = -np.ones_like(u)
        z = v
    else:
        raise ValueError(f"Unknown cube face: {face}. Must be one of {CUBE_FACE_KEYS}")

    return x, y, z


def build_e2c_maps(eq_width: int, eq_height: int, face_size: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Precomputes cv2.remap coordinate lookup tables for all 6 cube faces.
    """
    maps = {}
    for face in CUBE_FACE_KEYS:
        x, y, z = _build_e2c_face_rays(face, face_size)
        rho = np.sqrt(x * x + y * y + z * z)
        theta = np.arctan2(x, z)          # [-pi, pi]
        phi = np.arcsin(np.clip(y / rho, -1.0, 1.0))  # [-pi/2, pi/2]

        # Map to equirectangular pixel space
        map_x = ((theta / np.pi + 1.0) * 0.5 * eq_width - 0.5).astype(np.float32)
        map_y = ((0.5 - phi / np.pi) * eq_height - 0.5).astype(np.float32)

        # Wrap horizontal map_x cleanly around [0, eq_width)
        map_x = np.mod(map_x, eq_width)

        maps[face] = (map_x, map_y)
    return maps


def equirectangular_to_cubemap(
    image: np.ndarray,
    face_size: int = 512,
    is_mask: bool = False,
    cached_maps: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Converts a 360° equirectangular image or mask to 6 cube faces.

    Args:
        image: Numpy array of shape (H, W, 3) or (H, W).
        face_size: Output resolution of each square face.
        is_mask: If True, uses nearest neighbor interpolation and thresholding.
        cached_maps: Optional precomputed remap coordinates for performance.

    Returns:
        Dict mapping face name ('F', 'R', 'B', 'L', 'U', 'D') to square numpy array (face_size, face_size, ...).
    """
    eq_h, eq_w = image.shape[:2]
    maps = cached_maps or build_e2c_maps(eq_w, eq_h, face_size)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    faces = {}

    for face in CUBE_FACE_KEYS:
        map_x, map_y = maps[face]
        face_img = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=interp,
            borderMode=cv2.BORDER_WRAP,
        )
        if is_mask:
            face_img = (face_img > 127).astype(np.uint8) * 255
        faces[face] = face_img

    return faces


def build_c2e_maps(eq_width: int, eq_height: int, face_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precomputes lookup maps for cubemap to equirectangular projection.
    Returns:
        face_idx_map: (eq_height, eq_width) integer array [0..5] indicating which cube face each pixel maps to.
        map_x: (eq_height, eq_width) float32 x-coordinate within the selected face.
        map_y: (eq_height, eq_width) float32 y-coordinate within the selected face.
    """
    col, row = np.meshgrid(
        np.arange(eq_width, dtype=np.float32),
        np.arange(eq_height, dtype=np.float32),
    )

    theta = (col / eq_width) * 2.0 * np.pi - np.pi
    phi = np.pi * 0.5 - (row / eq_height) * np.pi

    cos_phi = np.cos(phi)
    x = cos_phi * np.sin(theta)
    y = np.sin(phi)
    z = cos_phi * np.cos(theta)

    abs_x = np.abs(x)
    abs_y = np.abs(y)
    abs_z = np.abs(z)

    max_axis = np.maximum(np.maximum(abs_x, abs_y), abs_z)

    face_idx = np.zeros((eq_height, eq_width), dtype=np.int32)
    u = np.zeros((eq_height, eq_width), dtype=np.float32)
    v = np.zeros((eq_height, eq_width), dtype=np.float32)

    # Masks for each face based on dominant axis
    eps = 1e-7
    # +X: Right (R, idx 1)
    mask_r = (abs_x == max_axis) & (x > 0)
    # -X: Left (L, idx 3)
    mask_l = (abs_x == max_axis) & (x <= 0)
    # +Y: Up (U, idx 4)
    mask_u = (abs_y == max_axis) & (y > 0)
    # -Y: Down (D, idx 5)
    mask_d = (abs_y == max_axis) & (y <= 0)
    # +Z: Front (F, idx 0)
    mask_f = (abs_z == max_axis) & (z > 0)
    # -Z: Back (B, idx 2)
    mask_b = (abs_z == max_axis) & (z <= 0)

    # Front
    face_idx[mask_f] = 0
    denom = np.maximum(abs_z[mask_f], eps)
    u[mask_f] = x[mask_f] / denom
    v[mask_f] = y[mask_f] / denom

    # Right
    face_idx[mask_r] = 1
    denom = np.maximum(abs_x[mask_r], eps)
    u[mask_r] = -z[mask_r] / denom
    v[mask_r] = y[mask_r] / denom

    # Back
    face_idx[mask_b] = 2
    denom = np.maximum(abs_z[mask_b], eps)
    u[mask_b] = -x[mask_b] / denom
    v[mask_b] = y[mask_b] / denom

    # Left
    face_idx[mask_l] = 3
    denom = np.maximum(abs_x[mask_l], eps)
    u[mask_l] = z[mask_l] / denom
    v[mask_l] = y[mask_l] / denom

    # Up
    face_idx[mask_u] = 4
    denom = np.maximum(abs_y[mask_u], eps)
    u[mask_u] = x[mask_u] / denom
    v[mask_u] = -z[mask_u] / denom

    # Down
    face_idx[mask_d] = 5
    denom = np.maximum(abs_y[mask_d], eps)
    u[mask_d] = x[mask_d] / denom
    v[mask_d] = z[mask_d] / denom

    # Convert normalized (u, v) in [-1, 1] to face pixel coordinates in [0, face_size - 1]
    map_x = np.clip((u + 1.0) * 0.5 * face_size - 0.5, 0, face_size - 1).astype(np.float32)
    map_y = np.clip((1.0 - v) * 0.5 * face_size - 0.5, 0, face_size - 1).astype(np.float32)

    return face_idx, map_x, map_y


def cubemap_to_equirectangular(
    faces: Dict[str, np.ndarray],
    out_width: int,
    out_height: int,
    is_mask: bool = False,
    cached_maps: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """
    Stitches 6 cube faces back into an equirectangular panorama.

    Args:
        faces: Dict mapping 'F', 'R', 'B', 'L', 'U', 'D' to square numpy arrays of equal size.
        out_width: Output equirectangular width.
        out_height: Output equirectangular height.
        is_mask: If True, uses nearest-neighbor interpolation.
        cached_maps: Precomputed (face_idx, map_x, map_y) from build_c2e_maps.

    Returns:
        Equirectangular numpy array of shape (out_height, out_width, 3) or (out_height, out_width).
    """
    first_face = faces["F"]
    face_size = first_face.shape[0]
    is_rgb = (first_face.ndim == 3)

    face_idx_map, map_x, map_y = cached_maps or build_c2e_maps(out_width, out_height, face_size)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    face_list = [faces[k] for k in CUBE_FACE_KEYS]

    if is_rgb:
        out_pano = np.zeros((out_height, out_width, 3), dtype=first_face.dtype)
    else:
        out_pano = np.zeros((out_height, out_width), dtype=first_face.dtype)

    for idx, face_img in enumerate(face_list):
        region_mask = (face_idx_map == idx)
        if not np.any(region_mask):
            continue

        remapped_face = cv2.remap(
            face_img,
            map_x,
            map_y,
            interpolation=interp,
            borderMode=cv2.BORDER_REPLICATE,
        )

        out_pano[region_mask] = remapped_face[region_mask]

    if is_mask:
        out_pano = (out_pano > 127).astype(np.uint8) * 255

    return out_pano


def get_active_faces(face_masks: Dict[str, np.ndarray], min_mask_pixels: int = 10) -> List[str]:
    """
    Determines which cube faces contain segmented furniture and need inpainting.
    """
    active = []
    for face_key in CUBE_FACE_KEYS:
        if face_key in face_masks:
            mask = face_masks[face_key]
            if np.count_nonzero(mask) >= min_mask_pixels:
                active.append(face_key)
    return active
