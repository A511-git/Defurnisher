"""
SAM 3 segmentation with per-prompt mask extraction.
Utilizes native Meta SAM 3 image caching and text grounding.
"""

from pathlib import Path
import re
from typing import Any, List
import cv2
import numpy as np
from PIL import Image

from ..config.schema import SAMConfig
from ..core.logger import get_logger


def sanitize_prompt_filename(prompt: str) -> str:
    """
    Sanitizes prompt string into a safe filename (e.g. 'coffee table' -> 'coffee_table').
    """
    clean = prompt.strip().lower()
    clean = re.sub(r"\s+", "_", clean)
    clean = re.sub(r"[^\w\-]", "", clean)
    return clean or "unnamed_mask"


def extract_binary_mask_from_state(state: dict, height: int, width: int) -> np.ndarray:
    """
    Extracts union binary mask (H, W) uint8 {0, 255} from processor inference state.
    """
    combined = np.zeros((height, width), dtype=np.uint8)
    if "masks" not in state or state["masks"] is None:
        return combined

    masks_tensor = state["masks"]
    if hasattr(masks_tensor, "numel") and masks_tensor.numel() == 0:
        return combined

    try:
        # masks_tensor has shape (N, 1, H, W)
        masks_np = masks_tensor.squeeze(1).detach().cpu().numpy()
        if masks_np.ndim == 2:
            masks_np = masks_np[np.newaxis, ...]

        for m in masks_np:
            binary_m = (m > 0).astype(np.uint8) * 255
            combined = cv2.bitwise_or(combined, binary_m)
    except Exception as e:
        get_logger().debug(f"Error extracting mask from tensor: {e}")

    return combined


def run_multi_pass_segmentation(
    model: Any,
    processor: Any,
    image_path: Path,
    masks_output_dir: Path,
    config: SAMConfig,
) -> np.ndarray:
    """
    Runs SAM 3 text-grounded segmentation for all configured furniture prompts:
    1. Encodes the image once via processor.set_image(image_pil).
    2. Sets confidence threshold once on the cached state.
    3. Queries each prompt in config.prompts individually.
    4. Extracts and saves individual binary masks (e.g. chair.png, bed.png).
    5. Computes boolean union mask, excluding prompts specified in config.skip_prompts.

    Returns:
        Unified binary mask (H, W) uint8 {0, 255}.
    """
    logger = get_logger()
    logger.info(f"Starting SAM 3 segmentation for: {image_path.name} ({len(config.prompts)} prompts)")

    masks_output_dir.mkdir(parents=True, exist_ok=True)

    image_pil = Image.open(image_path).convert("RGB")
    width, height = image_pil.size
    union_mask = np.zeros((height, width), dtype=np.uint8)

    # Prepare normalized set of skipped prompts
    skip_set = {p.strip().lower() for p in (config.skip_prompts or [])}

    # Cache image in processor (visual backbone runs ONCE)
    state = processor.set_image(image_pil)

    # Set confidence cutoff
    if hasattr(processor, "set_confidence_threshold"):
        processor.set_confidence_threshold(config.threshold, state)

    # Deduplicate prompts preserving order
    unique_prompts = list(dict.fromkeys(config.prompts))

    try:
        for prompt in unique_prompts:
            clean_name = sanitize_prompt_filename(prompt)
            prompt_mask = np.zeros((height, width), dtype=np.uint8)

            try:
                state = processor.set_text_prompt(prompt, state)
                prompt_mask = extract_binary_mask_from_state(state, height, width)
                processor.reset_all_prompts(state)
            except Exception as e:
                logger.debug(f"SAM 3 text prompt '{prompt}' query notice: {e}")

            # Save individual prompt mask (e.g. chair.png, bed.png)
            mask_filename = masks_output_dir / f"{clean_name}.png"
            cv2.imwrite(str(mask_filename), prompt_mask)

            # Check if this prompt should be excluded from final union
            prompt_norm = prompt.strip().lower()
            if prompt_norm in skip_set or clean_name in skip_set:
                logger.info(f"Saved mask '{clean_name}.png', excluded from final union (--sam-skip-prompt)")
            else:
                union_mask = cv2.bitwise_or(union_mask, prompt_mask)

    finally:
        # Free processor image state
        if hasattr(processor, "reset_all_prompts"):
            processor.reset_all_prompts(state)
        del state

    logger.info(f"Completed SAM 3 segmentation for {image_path.name}.")
    return union_mask
