"""
Unit tests for path management and job discovery.
"""

import tempfile
import unittest
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from defurnishing.core.path_manager import discover_jobs


class TestPathManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.input_dir = self.base_path / "input_panos"
        self.input_dir.mkdir()
        self.output_dir = self.base_path / "output"

        # Create dummy test images
        (self.input_dir / "room1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
        (self.input_dir / "room2.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        # Create non-image file that should be ignored
        (self.input_dir / "notes.txt").write_text("not an image")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_single_file_job_discovery(self):
        """Validates job structure when pointing to a single image file."""
        single_file = self.input_dir / "room1.jpg"
        jobs = discover_jobs(str(single_file), str(self.output_dir))

        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job.image_path, single_file)
        self.assertEqual(job.output_dir, self.output_dir / "room1")
        self.assertEqual(job.masks_dir, self.output_dir / "room1" / "masks")
        self.assertEqual(job.intermediate_dir, self.output_dir / "room1" / "intermediate_outputs")
        self.assertEqual(job.final_output_path, self.output_dir / "room1" / "final_output.jpg")
        self.assertTrue(job.masks_dir.exists())
        self.assertTrue(job.intermediate_dir.exists())

    def test_directory_job_discovery(self):
        """Validates job structure when pointing to a folder containing images."""
        jobs = discover_jobs(str(self.input_dir), str(self.output_dir))

        self.assertEqual(len(jobs), 2)
        stems = {j.image_path.stem for j in jobs}
        self.assertEqual(stems, {"room1", "room2"})

        for job in jobs:
            self.assertTrue(job.masks_dir.exists())
            self.assertTrue(job.intermediate_dir.exists())


if __name__ == "__main__":
    unittest.main()
