"""
Unit tests for Equirectangular <-> Cubemap geometry transformations.
"""

import sys
from pathlib import Path
import unittest
import numpy as np

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from defurnishing.geometry.cubemap import (
    CUBE_FACE_KEYS,
    equirectangular_to_cubemap,
    cubemap_to_equirectangular,
    get_active_faces,
)


class TestCubemapTransforms(unittest.TestCase):

    def setUp(self):
        self.eq_h = 256
        self.eq_w = 512
        self.face_size = 128

        # Create synthetic equirectangular image with gradients
        y, x = np.mgrid[0:self.eq_h, 0:self.eq_w]
        self.test_img = np.stack([
            (x % 256).astype(np.uint8),
            (y % 256).astype(np.uint8),
            ((x + y) % 256).astype(np.uint8),
        ], axis=-1)

        # Create mask at center (front face)
        self.test_mask = np.zeros((self.eq_h, self.eq_w), dtype=np.uint8)
        # Center of equirectangular (Front face is around x: 256 +/- 64, y: 128 +/- 64)
        self.test_mask[100:156, 220:292] = 255

    def test_e2c_faces_generated(self):
        """Verifies that all 6 cube faces are generated with correct square dimensions."""
        faces = equirectangular_to_cubemap(self.test_img, face_size=self.face_size)
        self.assertEqual(set(faces.keys()), set(CUBE_FACE_KEYS))
        for key in CUBE_FACE_KEYS:
            self.assertEqual(faces[key].shape, (self.face_size, self.face_size, 3))

    def test_mask_e2c_binary(self):
        """Verifies that projected mask faces remain binary (0 or 255)."""
        mask_faces = equirectangular_to_cubemap(self.test_mask, face_size=self.face_size, is_mask=True)
        for key in CUBE_FACE_KEYS:
            unique_vals = np.unique(mask_faces[key])
            for v in unique_vals:
                self.assertIn(v, [0, 255])

    def test_active_faces_detection(self):
        """Verifies that get_active_faces correctly identifies the front face containing the mask."""
        mask_faces = equirectangular_to_cubemap(self.test_mask, face_size=self.face_size, is_mask=True)
        active = get_active_faces(mask_faces, min_mask_pixels=50)
        self.assertIn("F", active)
        # Back face should not have the mask
        self.assertNotIn("B", active)

    def test_roundtrip_dimensions(self):
        """Verifies e2c followed by c2e restores the original dimensions."""
        faces = equirectangular_to_cubemap(self.test_img, face_size=self.face_size)
        restored = cubemap_to_equirectangular(faces, out_width=self.eq_w, out_height=self.eq_h)
        self.assertEqual(restored.shape, self.test_img.shape)

    def test_mask_roundtrip(self):
        """Verifies mask roundtrip preserves binary mask representation."""
        mask_faces = equirectangular_to_cubemap(self.test_mask, face_size=self.face_size, is_mask=True)
        restored_mask = cubemap_to_equirectangular(mask_faces, out_width=self.eq_w, out_height=self.eq_h, is_mask=True)
        self.assertEqual(restored_mask.shape, self.test_mask.shape)
        self.assertTrue(np.all((restored_mask == 0) | (restored_mask == 255)))
        # Restored mask center should overlap heavily with original mask center
        overlap = np.logical_and(restored_mask > 0, self.test_mask > 0)
        self.assertGreater(np.sum(overlap), 100)

    def test_down_face_floor_mask(self):
        """Verifies that a mask on the floor (nadir) is correctly projected to the Down face."""
        floor_mask = np.zeros((self.eq_h, self.eq_w), dtype=np.uint8)
        # Nadir is at bottom row
        floor_mask[220:256, :] = 255
        mask_faces = equirectangular_to_cubemap(floor_mask, face_size=self.face_size, is_mask=True)
        active = get_active_faces(mask_faces, min_mask_pixels=50)
        self.assertIn("D", active)


if __name__ == "__main__":
    unittest.main()
