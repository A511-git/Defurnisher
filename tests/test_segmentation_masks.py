"""
Unit tests for SAM 3 per-prompt segmentation mask generation, prompt sanitization,
and --sam-skip-prompt union exclusion.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from defurnishing.config.schema import SAMConfig
from defurnishing.segmentation.multi_pass import (
    extract_binary_mask_from_state,
    run_multi_pass_segmentation,
    sanitize_prompt_filename,
)


class MockTensor:
    """Mock tensor supporting squeeze, detach, cpu, numpy, numel."""
    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def squeeze(self, dim: int):
        return MockTensor(np.squeeze(self.arr, axis=dim))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr

    def numel(self):
        return self.arr.size


class MockSamProcessor:
    """Mock SAM 3 Processor simulating state management and text queries."""
    def __init__(self, prompt_mask_map=None):
        self.prompt_mask_map = prompt_mask_map or {}
        self.confidence_threshold = 0.4
        self.reset_called = False

    def set_image(self, image_pil):
        return {"image_size": image_pil.size, "backbone_out": {}}

    def set_confidence_threshold(self, threshold: float, state=None):
        self.confidence_threshold = threshold
        return state

    def set_text_prompt(self, prompt: str, state: dict):
        if prompt in self.prompt_mask_map:
            mask_arr = self.prompt_mask_map[prompt]
            # Expected shape in SAM 3 state is (N, 1, H, W)
            tensor_arr = mask_arr[np.newaxis, np.newaxis, ...]
            state["masks"] = MockTensor(tensor_arr)
        else:
            state["masks"] = None
        return state

    def reset_all_prompts(self, state: dict):
        self.reset_called = True
        state.pop("masks", None)


class TestSegmentationMasks(unittest.TestCase):

    def test_sanitize_prompt_filename(self):
        self.assertEqual(sanitize_prompt_filename("chair"), "chair")
        self.assertEqual(sanitize_prompt_filename("coffee table"), "coffee_table")
        self.assertEqual(sanitize_prompt_filename(" TV Stand! "), "tv_stand")
        self.assertEqual(sanitize_prompt_filename("bed-side table"), "bed-side_table")
        self.assertEqual(sanitize_prompt_filename("   "), "unnamed_mask")

    def test_extract_binary_mask_empty_state(self):
        mask = extract_binary_mask_from_state({}, 100, 200)
        self.assertEqual(mask.shape, (100, 200))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(np.count_nonzero(mask), 0)

        mask_none = extract_binary_mask_from_state({"masks": None}, 50, 50)
        self.assertEqual(np.count_nonzero(mask_none), 0)

    def test_extract_binary_mask_with_tensor(self):
        # Create a mock 1-object mask of size (1, 1, 10, 10)
        raw = np.zeros((1, 1, 10, 10), dtype=np.float32)
        raw[0, 0, 2:6, 2:6] = 1.0  # 4x4 positive box
        state = {"masks": MockTensor(raw)}

        mask = extract_binary_mask_from_state(state, 10, 10)
        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(np.count_nonzero(mask), 16)
        self.assertEqual(mask[2, 2], 255)
        self.assertEqual(mask[0, 0], 0)

    def test_run_multi_pass_segmentation_saves_per_prompt_and_respects_skip(self):
        # Setup mock masks for "chair", "bed", "coffee table"
        h, w = 40, 60
        chair_mask = np.zeros((h, w), dtype=np.float32)
        chair_mask[5:15, 5:15] = 1.0

        bed_mask = np.zeros((h, w), dtype=np.float32)
        bed_mask[20:30, 20:30] = 1.0

        table_mask = np.zeros((h, w), dtype=np.float32)
        table_mask[10:20, 40:50] = 1.0

        mock_processor = MockSamProcessor({
            "chair": chair_mask,
            "bed": bed_mask,
            "coffee table": table_mask,
        })

        # Test within a temporary directory inside workspace
        temp_dir = SRC_PATH.parent / "tests" / "_temp_test_masks"
        temp_dir.mkdir(parents=True, exist_ok=True)
        img_path = temp_dir / "test_pano.png"
        masks_dir = temp_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create a dummy image
            dummy_img = Image.new("RGB", (w, h), color=(128, 128, 128))
            dummy_img.save(img_path)

            config = SAMConfig(
                prompts=["chair", "bed", "coffee table"],
                skip_prompts=["bed"],  # "bed" should be saved, but NOT added to union_mask
                threshold=0.4,
            )

            union_mask = run_multi_pass_segmentation(
                model=None,
                processor=mock_processor,
                image_path=img_path,
                masks_output_dir=masks_dir,
                config=config,
            )

            # 1. Verify individual prompt masks were saved
            self.assertTrue((masks_dir / "chair.png").exists(), "chair.png was not saved")
            self.assertTrue((masks_dir / "bed.png").exists(), "bed.png was not saved")
            self.assertTrue((masks_dir / "coffee_table.png").exists(), "coffee_table.png was not saved")

            # 2. Verify run_1.png / run_2.png were NOT created
            self.assertFalse((masks_dir / "run_1.png").exists(), "run_1.png should not exist")
            self.assertFalse((masks_dir / "run_2.png").exists(), "run_2.png should not exist")

            # 3. Verify union_mask has pixels for chair and coffee_table, but NOT bed
            self.assertEqual(union_mask[7, 7], 255, "chair should be in union_mask")
            self.assertEqual(union_mask[15, 45], 255, "coffee_table should be in union_mask")
            self.assertEqual(union_mask[25, 25], 0, "bed should be EXCLUDED from union_mask due to skip_prompts")

            # 4. Verify bed.png file on disk actually contains bed mask pixels
            import cv2
            saved_bed = cv2.imread(str(masks_dir / "bed.png"), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(saved_bed)
            self.assertEqual(saved_bed[25, 25], 255, "bed.png on disk should have detected pixels")

        finally:
            # Cleanup
            import shutil
            if temp_dir.exists():
                shutil.rmtree(str(temp_dir), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
