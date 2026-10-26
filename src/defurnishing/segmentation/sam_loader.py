"""
SAM 3.1 Model Loader and VRAM Lifecycle Manager.
Leverages the native Meta SAM 3 model builder and Sam3Processor from dependencies/sam3.
"""

import gc
import os
from typing import Any, Tuple
import torch

from ..config.schema import SAMConfig
from ..core.env_setup import get_sam3_bpe_path, setup_dependencies
from ..core.logger import get_logger


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Maps string dtype identifier to torch.dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def load_sam_model(config: SAMConfig) -> Tuple[Any, Any]:
    """
    Loads native SAM 3 model and processor into VRAM using configured precision.

    Args:
        config: SAMConfig instance.

    Returns:
        Tuple of (model, processor).
    """
    logger = get_logger()
    setup_dependencies()

    logger.info(
        f"Loading native SAM 3 from '{config.model_path}' on device '{config.device}' (dtype={config.dtype})..."
    )

    # Authenticate with Hugging Face Hub if token is provided (for gated model access)
    if config.hf_key:
        try:
            from huggingface_hub import login
            login(token=config.hf_key, add_to_git_credential=False)
            logger.info("Authenticated with Hugging Face Hub.")
        except Exception as e:
            logger.debug(f"HF login notice: {e}")

    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        torch_dtype = get_torch_dtype(config.dtype)
        device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")

        # Resolve local BPE vocabulary asset from dependencies/sam3
        bpe_path = get_sam3_bpe_path()
        if bpe_path:
            logger.debug(f"Using SAM 3 BPE tokenizer vocabulary from: {bpe_path}")

        # Build native SAM 3 image model
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            eval_mode=True,
            load_from_HF=True,
        )

        if hasattr(model, "to") and config.device != "cpu":
            # Autocast / dtype
            try:
                model = model.to(dtype=torch_dtype)
            except Exception as e:
                logger.debug(f"Model precision conversion note: {e}")

        # Instantiate native Sam3Processor
        processor = Sam3Processor(
            model=model,
            device=device,
            confidence_threshold=config.threshold,
        )

        logger.info("Native SAM 3 model and Sam3Processor successfully loaded into memory.")
        return model, processor

    except ImportError as e:
        logger.error(
            f"Failed to import native SAM 3: {e}. "
            "Please ensure dependencies/sam3 exists and required packages are installed."
        )
        raise
    except Exception as e:
        logger.error(f"Error initializing SAM 3 model: {e}")
        raise


def unload_sam_model(model: Any, processor: Any) -> None:
    """
    Explicitly deletes SAM 3 model references and flushes CUDA cache to free VRAM for diffusion.
    """
    logger = get_logger()
    logger.info("Unloading SAM 3 from VRAM...")

    del model
    del processor
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    logger.info("SAM 3 unloaded and CUDA cache cleared.")
