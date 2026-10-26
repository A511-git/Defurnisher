"""
Stable Diffusion Inpainting Pipeline Loader and VRAM Lifecycle Manager.
Leverages the native Hugging Face Diffusers pipeline and schedulers from dependencies/diffusers.
"""

import gc
from typing import Any
import torch

from ..config.schema import SDConfig
from ..core.env_setup import setup_dependencies
from ..core.logger import get_logger
from ..segmentation.sam_loader import get_torch_dtype


def configure_scheduler(pipe: Any, scheduler_name: str) -> None:
    """Configures the denoising scheduler on the Diffusers pipeline using native classes."""
    logger = get_logger()
    name = scheduler_name.lower()

    try:
        if name in ("dpm++", "dpmsolver"):
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True
            )
            logger.debug("Configured native DPMSolverMultistepScheduler (Karras).")
        elif name in ("euler_a", "eulera"):
            from diffusers import EulerAncestralDiscreteScheduler
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            logger.debug("Configured native EulerAncestralDiscreteScheduler.")
        elif name == "ddim":
            from diffusers import DDIMScheduler
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            logger.debug("Configured native DDIMScheduler.")
        elif name == "pndm":
            from diffusers import PNDMScheduler
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
            logger.debug("Configured native PNDMScheduler.")
        else:
            logger.warning(f"Unrecognized scheduler '{scheduler_name}', retaining default.")
    except Exception as e:
        logger.warning(f"Failed to switch scheduler to '{scheduler_name}': {e}. Using default.")


def load_sd_pipeline(config: SDConfig) -> Any:
    """
    Loads native Stable Diffusion Inpainting pipeline into VRAM.

    Args:
        config: SDConfig instance.

    Returns:
        Configured StableDiffusionInpaintPipeline instance.
    """
    logger = get_logger()
    setup_dependencies()

    logger.info(
        f"Loading Stable Diffusion Inpainting from '{config.model_path}' "
        f"on device '{config.device}' (dtype={config.dtype}, scheduler={config.scheduler})..."
    )

    try:
        import diffusers.models
        from diffusers import StableDiffusionInpaintPipeline

        torch_dtype = get_torch_dtype(config.dtype)
        device = "cuda" if (torch.cuda.is_available() and config.device != "cpu") else "cpu"

        # Configure authentication token for gated model downloads if provided
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "safety_checker": None,
        }
        if config.hf_key:
            load_kwargs["token"] = config.hf_key

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            config.model_path,
            **load_kwargs,
        )
        pipe = pipe.to(device)

        configure_scheduler(pipe, config.scheduler)

        # Native Diffusers VRAM optimizations
        if config.attention_slicing:
            pipe.enable_attention_slicing()
            logger.info("Diffusers attention slicing enabled.")

        # VAE slicing/tiling reduces memory spikes on panoramas
        if hasattr(pipe, "enable_vae_slicing"):
            try:
                pipe.enable_vae_slicing()
                logger.debug("Diffusers VAE slicing enabled.")
            except Exception as e:
                logger.debug(f"VAE slicing notice: {e}")

        logger.info("Native Stable Diffusion Inpainting pipeline loaded successfully.")
        return pipe

    except Exception as e:
        logger.error(f"Failed to load Stable Diffusion Inpainting pipeline: {e}")
        raise


def unload_sd_pipeline(pipe: Any) -> None:
    """Explicitly deallocates Stable Diffusion from VRAM."""
    logger = get_logger()
    logger.info("Unloading Stable Diffusion pipeline from VRAM...")

    del pipe
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    logger.info("Stable Diffusion unloaded and CUDA cache cleared.")
