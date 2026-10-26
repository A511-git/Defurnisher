"""
Unit tests for equirectangular 360° geometry transformations.
"""

import unittest
import numpy as np

import sys
from pathlib import Path
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from defurnishing.geometry.roll import (
    compute_mask_centroid,
    calculate_roll_shift,
    roll_equirectangular,
    unroll_equirectangular,
)
from defurnishing.geometry.padding import (
    compute_pad_width,
    wrap_pad_equirectangular,
    unpad_equirectangular,
)
from defurnishing.geometry.resize import (
    compute_inpaint_dimensions,
)
from defurnishing.geometry.transform import (
    forward_geometry,
    inverse_geometry,
)
from defurnishing.config.schema import GeometryConfig


class TestGeometryTransforms(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 3-channel test pattern with distinctive gradients
        self.height = 256
        self.width = 512
        y, x = np.mgrid[0:self.height, 0:self.width]
        self.test_image = np.stack([
            (x % 256).astype(np.uint8),
            (y % 256).astype(np.uint8),
            ((x + y) % 256).astype(np.uint8),
        ], axis=-1)

        # Synthetic binary mask with a localized rectangle
        self.test_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.test_mask[50:150, 100:200] = 255

    def test_roll_unroll_roundtrip(self):
        """Validates that roll followed by unroll produces exact 100% pixel equality."""
        shift = 137
        rolled = roll_equirectangular(self.test_image, shift)
        unrolled = unroll_equirectangular(rolled, shift)
        self.assertTrue(np.array_equal(self.test_image, unrolled))

    def test_pad_unpad_roundtrip(self):
        """Validates that circular wrap-pad followed by unpad produces exact 100% pixel equality."""
        pad_width = compute_pad_width(self.width, pad_ratio=0.20)
        self.assertEqual(pad_width, int(512 * 0.20))

        padded = wrap_pad_equirectangular(self.test_image, pad_width)
        self.assertEqual(padded.shape[1], self.width + 2 * pad_width)

        unpadded = unpad_equirectangular(padded, pad_width, self.width)
        self.assertTrue(np.array_equal(self.test_image, unpadded))

    def test_mask_centroid(self):
        """Validates mask centroid calculation on known coordinates."""
        centroid = compute_mask_centroid(self.test_mask)
        # Expected x-center is mean(100..199) = 149.5
        self.assertAlmostEqual(centroid, 149.5, places=1)

        shift = calculate_roll_shift(self.test_mask, self.width)
        # Shift to center at 256: floor(256 - 149.5) = 106
        self.assertEqual(shift, 106)

    def test_dimensions_multiple_of_8(self):
        """Ensures dimensions generated for VAE latents are strictly divisible by 8."""
        for test_h in [500, 512, 768, 1024]:
            for test_w in [1000, 1024, 1536, 2048]:
                w, h = compute_inpaint_dimensions(test_w, test_h, target_height=512)
                self.assertEqual(w % 8, 0, f"Width {w} is not divisible by 8")
                self.assertEqual(h % 8, 0, f"Height {h} is not divisible by 8")

    def test_forward_inverse_pipeline_without_scaling(self):
        """Validates end-to-end forward and inverse pipeline geometry when target_height matches."""
        cfg = GeometryConfig(roll=True, pad=True, pad_ratio=0.20, target_height=self.height)
        result = forward_geometry(self.test_image, self.test_mask, cfg)

        # Invert geometry on the padded image directly (simulating 100% faithful inpaint)
        # Scale back to padded dimensions
        restored = inverse_geometry(result.transformed_image, result.to_dict())

        # Shapes must match exactly
        self.assertEqual(restored.shape, self.test_image.shape)


if __name__ == "__main__":
    unittest.main()
