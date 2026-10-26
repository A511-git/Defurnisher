"""
Job discovery and output directory tree management.
Produces standardized output structures matching the pipeline specification.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .logger import get_logger

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class JobPaths:
    """
    Encapsulates all input and output paths for a single panorama.
    """
    image_path: Path
    output_dir: Path
    masks_dir: Path
    intermediate_dir: Path
    final_output_path: Path

    def ensure_directories(self) -> None:
        """Creates the masks and intermediate_outputs directories if they do not exist."""
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)


def discover_jobs(
    input_path_str: str,
    output_root_str: str,
    recursive: bool = False,
    output_format: str = "jpg",
) -> List[JobPaths]:
    """
    Discovers all candidate panorama images from a single file or directory input.

    Args:
        input_path_str: Path to an image file or a directory containing images.
        output_root_str: Destination directory where output folders will be created.
        recursive: Whether to scan subdirectories if input is a directory.
        output_format: Extension for the final output image (e.g. 'jpg', 'png').

    Returns:
        List of initialized JobPaths with output directory structure.
    """
    logger = get_logger()
    input_path = Path(input_path_str).resolve()
    output_root = Path(output_root_str).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    jobs: List[JobPaths] = []
    clean_fmt = output_format.lstrip(".").lower()

    if input_path.is_file():
        # Case A: Single File Input
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension '{input_path.suffix}'. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )
        job_dir = output_root / input_path.stem
        job = JobPaths(
            image_path=input_path,
            output_dir=job_dir,
            masks_dir=job_dir / "masks",
            intermediate_dir=job_dir / "intermediate_outputs",
            final_output_path=job_dir / f"final_output.{clean_fmt}",
        )
        job.ensure_directories()
        jobs.append(job)
        logger.info(f"Discovered 1 single image job: {input_path.name}")
        return jobs

    # Case B: Directory Input
    logger.info(f"Scanning directory: {input_path} (recursive={recursive})")
    pattern = "**/*" if recursive else "*"
    all_files = [p for p in input_path.glob(pattern) if p.is_file()]
    image_files = [p for p in all_files if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not image_files:
        logger.warning(f"No supported images found in: {input_path}")
        return []

    input_parent_name = input_path.name

    for img_path in sorted(image_files):
        # Calculate relative path from input directory to preserve folder hierarchy if recursive
        rel_path = img_path.relative_to(input_path)
        # Output structure: output_root / input_folder_name / rel_subpath_stem /
        if len(rel_path.parts) > 1:
            rel_parent = rel_path.parent
            job_dir = output_root / input_parent_name / rel_parent / img_path.stem
        else:
            job_dir = output_root / input_parent_name / img_path.stem

        job = JobPaths(
            image_path=img_path,
            output_dir=job_dir,
            masks_dir=job_dir / "masks",
            intermediate_dir=job_dir / "intermediate_outputs",
            final_output_path=job_dir / f"final_output.{clean_fmt}",
        )
        job.ensure_directories()
        jobs.append(job)

    logger.info(f"Discovered {len(jobs)} panorama image jobs.")
    return jobs
