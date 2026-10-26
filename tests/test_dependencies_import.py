"""
Unit tests for verifying native repository imports for SAM 3 and Hugging Face Diffusers.
"""

import sys
import unittest
from pathlib import Path

# Add project 'src' directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from defurnishing.core.env_setup import (
    get_sam3_bpe_path,
    get_sam3_root,
    setup_dependencies,
    verify_dependencies,
)


class TestDependenciesImport(unittest.TestCase):

    def setUp(self):
        self.resolved = setup_dependencies()

    def test_local_directories_discovered(self):
        """Verifies that dependencies/sam3 and dependencies/diffusers/src are found."""
        self.assertIn("sam3", self.resolved)
        self.assertIn("diffusers", self.resolved)

        sam3_path = Path(self.resolved["sam3"])
        self.assertTrue(sam3_path.is_dir(), f"SAM3 path not a directory: {sam3_path}")

        diffusers_path = Path(self.resolved["diffusers"])
        self.assertTrue(diffusers_path.is_dir(), f"Diffusers path not a directory: {diffusers_path}")

    def test_bpe_vocab_asset_exists(self):
        """Verifies that the SAM 3 BPE tokenizer vocabulary file exists."""
        bpe_path = get_sam3_bpe_path()
        self.assertIsNotNone(bpe_path, "Could not locate SAM 3 BPE vocabulary asset.")
        self.assertTrue(Path(bpe_path).is_file(), f"BPE file not found at: {bpe_path}")

    def test_sam3_symbols_importable(self):
        """Verifies that SAM 3 model builder and processor can be imported from dependencies/sam3."""
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.assertTrue(callable(build_sam3_image_model))
        self.assertTrue(callable(Sam3Processor))

    def test_diffusers_symbols_importable(self):
        """Verifies that Diffusers inpainting pipeline and schedulers can be imported."""
        from diffusers import (
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            StableDiffusionInpaintPipeline,
        )

        self.assertTrue(callable(StableDiffusionInpaintPipeline))
        self.assertTrue(callable(DPMSolverMultistepScheduler))
        self.assertTrue(callable(EulerAncestralDiscreteScheduler))


if __name__ == "__main__":
    unittest.main()
