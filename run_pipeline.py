"""
Top-level entry point for Matterport Panorama Defurnishing Pipeline.

Usage:
    # Direct hit using defaults from config.yaml:
    python run_pipeline.py --input ./panos --output ./output

    # With custom log level and SAM refinement passes:
    python run_pipeline.py --input ./panos --output ./output --log-level DEBUG --sam-refine 4

    # Run masks-only or inpaint-only:
    python run_pipeline.py --input ./panos --output ./output --stage masks-only
"""

import sys
from pathlib import Path

# Add 'src' directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from defurnishing.config.loader import load_config_and_args
from defurnishing.pipeline import run_pipeline


def main() -> None:
    config = load_config_and_args()
    run_pipeline(config)


if __name__ == "__main__":
    main()
