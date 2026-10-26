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
    import os
    hf_token = config.hf_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            logger.info("Authenticated with Hugging Face Hub.")
        except Exception as e:
            logger.debug(f"HF login notice: {e}")
    else:
        logger.warning(
            "No Hugging Face token detected (--hf-key or $HF_TOKEN). "
            "Gated models (e.g. facebook/sam3) require authentication."
        )

    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        torch_dtype = get_torch_dtype(config.dtype)
        device = "cuda" if (torch.cuda.is_available() and config.device != "cpu") else "cpu"

        # Resolve local BPE vocabulary asset from dependencies/sam3
        bpe_path = get_sam3_bpe_path()
        if bpe_path:
            logger.debug(f"Using SAM 3 BPE tokenizer vocabulary from: {bpe_path}")

        # Determine whether to load sam3.1 or sam3 based on config.model_path
        version = "sam3.1" if "3.1" in str(config.model_path) else "sam3"
        logger.info(f"Targeting SAM checkpoint version: '{version}' (repo: facebook/{version})")

        # Build native SAM 3 image model
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            eval_mode=True,
            load_from_HF=True,
            version=version,
        )

        # Explicitly ensure the entire model is on the target device in native precision
        if hasattr(model, "to"):
            model = model.to(device=device)
            logger.info(f"SAM 3 model successfully transferred to {device}.")

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
        err_msg = str(e)
        if "401" in err_msg or "gated" in err_msg.lower() or "restricted" in err_msg.lower():
            logger.error(
                f"Failed to access gated model '{config.model_path}' (HTTP 401).\n"
                "To resolve this:\n"
                f"  1. Visit https://huggingface.co/{config.model_path} in your browser and accept the model license/terms.\n"
                "  2. Make sure your Hugging Face User Access Token has 'Read' permission (https://huggingface.co/settings/tokens).\n"
                "  3. Pass the token via --hf-key or set os.environ['HF_TOKEN']."
            )
        else:
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
