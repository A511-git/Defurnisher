"""
Stage 3: Stable Diffusion Inpainting pipeline loading and batch inference.
"""

from .sd_loader import load_sd_pipeline, unload_sd_pipeline
from .inpaint_runner import run_inpaint_inference

__all__ = [
    "load_sd_pipeline",
    "unload_sd_pipeline",
    "run_inpaint_inference",
]
