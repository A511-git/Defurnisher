"""
Unit tests for configuration loader and CLI argument parsing.
"""

import unittest
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from defurnishing.config.loader import load_config_and_args


class TestConfigLoader(unittest.TestCase):

    def test_default_config_loading(self):
        """Verifies loading with default config.yaml."""
        config = load_config_and_args(args_list=["--input", "dummy.jpg"])
        self.assertEqual(config.input, "dummy.jpg")
        self.assertEqual(config.stage, "all")
        self.assertIn(config.log_level, ["DEBUG", "INFO", "WARNING", "ERROR"])
        self.assertEqual(config.sam.skip_prompts, [])
        self.assertEqual(config.sd.scheduler, "dpm++")

    def test_cli_overrides(self):
        """Verifies CLI arguments override configuration file defaults."""
        args = [
            "--input", "panos/test.png",
            "--output", "custom_output",
            "--log-level", "DEBUG",
            "--stage", "masks-only",
            "--sam-dtype", "float16",
            "--sam-skip-prompt", "bed",
            "--sam-skip-prompt", "sofa",
            "--sd-steps", "40",
            "--no-roll",
            "--no-pad",
        ]
        config = load_config_and_args(args_list=args)

        self.assertEqual(config.input, "panos/test.png")
        self.assertEqual(config.output, "custom_output")
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.stage, "masks-only")
        self.assertEqual(config.sam.dtype, "float16")
        self.assertEqual(config.sam.skip_prompts, ["bed", "sofa"])
        self.assertEqual(config.sd.steps, 40)
        self.assertFalse(config.geometry.roll)
        self.assertFalse(config.geometry.pad)


if __name__ == "__main__":
    unittest.main()
