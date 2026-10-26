"""
Environment bootstrapper for local dependency auto-discovery.
Automatically discovers and injects 'dependencies/diffusers/src' and
'dependencies/sam3' into sys.path and resolves local assets.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

from .logger import get_logger


def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    Assumes this file is located at <project_root>/src/defurnishing/core/env_setup.py.
    """
    return Path(__file__).resolve().parent.parent.parent.parent


def get_sam3_root() -> Path:
    """Returns absolute path to the dependencies/sam3 directory."""
    return get_project_root() / "dependencies" / "sam3"


def get_sam3_bpe_path() -> Optional[str]:
    """
    Returns the string path to the BPE vocabulary file needed by SAM 3.
    Looks in dependencies/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz.
    """
    sam3_root = get_sam3_root()
    bpe_path = sam3_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if bpe_path.is_file():
        return str(bpe_path.resolve())

    # Fallback to alternate asset location
    alt_bpe = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if alt_bpe.is_file():
        return str(alt_bpe.resolve())

    return None


def setup_dependencies(project_root: Optional[Path] = None) -> Dict[str, str]:
    """
    Discovers local repositories in 'dependencies/' and adds them to sys.path.

    Returns:
        Dict mapping dependency name ('diffusers', 'sam3') to its resolved path.
    """
    logger = get_logger()
    if project_root is None:
        project_root = get_project_root()

    dependencies_dir = project_root / "dependencies"
    resolved_paths: Dict[str, str] = {}

    # 1. Hugging Face Diffusers (package lives under dependencies/diffusers/src)
    diffusers_src = dependencies_dir / "diffusers" / "src"
    if diffusers_src.exists():
        diffusers_str = str(diffusers_src.resolve())
        if diffusers_str not in sys.path:
            sys.path.insert(0, diffusers_str)
            logger.debug(f"Added local diffusers to sys.path: {diffusers_str}")
        resolved_paths["diffusers"] = diffusers_str
    else:
        logger.warning(
            f"Local diffusers not found at {diffusers_src}. "
            "Falling back to system-installed diffusers package."
        )

    # 2. Meta SAM 3 (package lives under dependencies/sam3)
    sam3_root = dependencies_dir / "sam3"
    if sam3_root.exists():
        sam3_str = str(sam3_root.resolve())
        if sam3_str not in sys.path:
            sys.path.insert(0, sam3_str)
            logger.debug(f"Added local sam3 to sys.path: {sam3_str}")
        resolved_paths["sam3"] = sam3_str
    else:
        logger.warning(
            f"Local sam3 not found at {sam3_root}. "
            "Falling back to system-installed sam3 package."
        )

    return resolved_paths


def verify_dependencies() -> Dict[str, bool]:
    """
    Verifies that required native modules from dependencies/ can be cleanly imported.

    Returns:
        Dict of {dependency_name: status_boolean}
    """
    setup_dependencies()
    status = {"sam3": False, "diffusers": False}

    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        status["sam3"] = True
    except Exception as e:
        get_logger().debug(f"SAM 3 import verification notice: {e}")

    try:
        from diffusers import StableDiffusionInpaintPipeline
        status["diffusers"] = True
    except Exception as e:
        get_logger().debug(f"Diffusers import verification notice: {e}")

    return status
