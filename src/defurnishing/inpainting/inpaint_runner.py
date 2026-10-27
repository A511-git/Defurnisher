"""
Inpainting inference execution for prepared equirectangular panoramas.
"""

from typing import Any
import numpy as np
from PIL import Image
import torch

from ..config.schema import SDConfig
from ..core.logger import get_logger


def run_inpaint_inference(
    pipe: Any,
    image_prepared: np.ndarray,
    mask_prepared: np.ndarray,
    config: SDConfig,
) -> np.ndarray:
    """
    Executes diffusion inpainting on the prepared (rolled, padded, scaled) image & mask.

    Args:
        pipe: StableDiffusionInpaintPipeline instance.
        image_prepared: RGB array (H, W, 3) uint8.
        mask_prepared: Binary mask array (H, W) uint8 {0, 255}.
        config: SDConfig instance.

    Returns:
        Generated inpainting result as RGB numpy array (H, W, 3) uint8.
    """
    logger = get_logger()
    logger.info(
        f"Inpainting image of shape {image_prepared.shape} "
        f"({config.steps} steps, CFG={config.guidance_scale}, seed={config.seed})..."
    )

    # Convert to PIL Image
    pil_image = Image.fromarray(image_prepared).convert("RGB")
    pil_mask = Image.fromarray(mask_prepared).convert("L")
    inpaint_h, inpaint_w = image_prepared.shape[:2]

    device = next(pipe.unet.parameters()).device
    generator = None
    if config.seed is not None and config.seed >= 0:
        generator = torch.Generator(device=device).manual_seed(config.seed)

    # Execute inpainting pipeline
    with torch.inference_mode():
        result = pipe(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            height=inpaint_h,
            width=inpaint_w,
            strength=config.strength,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
        )

    output_pil = result.images[0]
    output_np = np.array(output_pil)
    logger.info("Inpainting inference completed.")
    return output_np
