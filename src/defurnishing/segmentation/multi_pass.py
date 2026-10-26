"""
Multi-pass refinement and negative prompt subtraction for robust furniture segmentation.
Utilizes native Meta SAM 3 image caching and text grounding.
"""

from pathlib import Path
from typing import Any, List
import cv2
import numpy as np
from PIL import Image

from ..config.schema import SAMConfig
from ..core.logger import get_logger


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


def run_inference_pass_native(
    processor: Any,
    state: dict,
    prompts: List[str],
    negative_prompts: List[str],
    threshold: float,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Runs text grounding queries against the cached image state using native Sam3Processor.
    """
    logger = get_logger()
    pos_mask = np.zeros((height, width), dtype=np.uint8)
    neg_mask = np.zeros((height, width), dtype=np.uint8)

    # Set confidence threshold for this pass
    if hasattr(processor, "set_confidence_threshold"):
        processor.set_confidence_threshold(threshold, state)

    # 1. Query positive furniture concepts
    for prompt in prompts:
        try:
            state = processor.set_text_prompt(prompt, state)
            prompt_mask = extract_binary_mask_from_state(state, height, width)
            pos_mask = cv2.bitwise_or(pos_mask, prompt_mask)
            processor.reset_all_prompts(state)
        except Exception as e:
            logger.debug(f"SAM 3 text prompt '{prompt}' query notice: {e}")

    # 2. Query negative structural concepts (walls, floors, ceilings)
    for neg_prompt in negative_prompts:
        try:
            state = processor.set_text_prompt(neg_prompt, state)
            prompt_mask = extract_binary_mask_from_state(state, height, width)
            neg_mask = cv2.bitwise_or(neg_mask, prompt_mask)
            processor.reset_all_prompts(state)
        except Exception as e:
            logger.debug(f"SAM 3 negative prompt '{neg_prompt}' query notice: {e}")

    # 3. Negative prompt subtraction: Pos AND NOT Neg
    cleaned_mask = cv2.bitwise_and(pos_mask, cv2.bitwise_not(neg_mask))
    return cleaned_mask


def run_multi_pass_segmentation(
    model: Any,
    processor: Any,
    image_path: Path,
    masks_output_dir: Path,
    config: SAMConfig,
) -> np.ndarray:
    """
    Runs multi-pass refinement (--sam-refine passes) using native SAM 3:
    1. Encodes the image once via processor.set_image(image_pil).
    2. Runs pass 1: Primary furniture prompts.
    3. Runs pass 2: Secondary furniture & decor prompts.
    4. Runs pass 3+: Sensitive passes with reduced threshold.
    5. Subtracts negative prompt masks and computes boolean union.

    Returns:
        Unified binary mask (H, W) uint8 {0, 255}.
    """
    logger = get_logger()
    logger.info(f"Starting native SAM 3 segmentation for: {image_path.name} ({config.refine} passes)")

    image_pil = Image.open(image_path).convert("RGB")
    width, height = image_pil.size
    union_mask = np.zeros((height, width), dtype=np.uint8)

    # Cache image in processor (visual backbone runs ONCE)
    state = processor.set_image(image_pil)

    # Partition prompts for multi-pass strategy
    mid = max(1, len(config.prompts) // 2)
    primary_prompts = config.prompts[:mid]
    secondary_prompts = config.prompts[mid:] if len(config.prompts) > mid else config.prompts

    try:
        for pass_idx in range(1, config.refine + 1):
            if pass_idx == 1:
                current_prompts = primary_prompts
                current_thresh = config.threshold
            elif pass_idx == 2:
                current_prompts = secondary_prompts
                current_thresh = config.threshold
            else:
                # Sensitive pass: all prompts with lower threshold (-0.10) to catch distant items
                current_prompts = config.prompts
                current_thresh = max(0.15, config.threshold - 0.10)

            logger.debug(f"Pass {pass_idx}/{config.refine}: threshold={current_thresh:.2f}, {len(current_prompts)} prompts")

            pass_mask = run_inference_pass_native(
                processor=processor,
                state=state,
                prompts=current_prompts,
                negative_prompts=config.negative_prompts,
                threshold=current_thresh,
                height=height,
                width=width,
            )

            # Save individual run mask
            pass_filename = masks_output_dir / f"run_{pass_idx}.png"
            cv2.imwrite(str(pass_filename), pass_mask)

            # Boolean union
            union_mask = cv2.bitwise_or(union_mask, pass_mask)

    finally:
        # Free processor image state
        if hasattr(processor, "reset_all_prompts"):
            processor.reset_all_prompts(state)
        del state

    logger.info(f"Completed {config.refine} native SAM 3 passes for {image_path.name}.")
    return union_mask
