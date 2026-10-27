"""
YAML configuration loader and CLI argument parser.
Allows zero-flag direct-hit execution while supporting granular CLI overrides.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .schema import GeometryConfig, PipelineConfig, SAMConfig, SDConfig


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Reads a YAML configuration file if it exists."""
    if not config_path.is_file():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the comprehensive CLI argument parser with prefixed options."""
    parser = argparse.ArgumentParser(
        description="Matterport Panorama Defurnishing Pipeline (CVPRW 2024)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General / Pipeline options
    gen = parser.add_argument_group("General Pipeline Options")
    gen.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to a single 360° panorama image or a directory of panoramas."
    )
    gen.add_argument(
        "--output", "-o", type=str, default=None,
        help="Root output directory."
    )
    gen.add_argument(
        "--config", "-c", type=str, default="config.yaml",
        help="Path to YAML configuration file."
    )
    gen.add_argument(
        "--stage", type=str, choices=["all", "masks-only", "inpaint-only"], default=None,
        help="Execution stage."
    )
    gen.add_argument(
        "--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None,
        help="Logging verbosity level."
    )
    gen.add_argument(
        "--hf-key", "--hf-token", type=str, default=None,
        help="Hugging Face API token / key for authenticating and downloading models."
    )
    gen.add_argument(
        "--recursive", action="store_true", default=None,
        help="Recursively scan subdirectories for input images."
    )
    gen.add_argument(
        "--overwrite", action="store_true", default=None,
        help="Overwrite existing output files."
    )

    # SAM 3.1 Options
    sam_grp = parser.add_argument_group("Meta SAM 3.1 Segmentation Options (--sam-*)")
    sam_grp.add_argument(
        "--sam-model-path", type=str, default=None,
        help="SAM 3.1 HuggingFace repository ID or local weights path."
    )
    sam_grp.add_argument(
        "--sam-device", type=str, default=None,
        help="Device for SAM 3.1 inference ('cuda' or 'cpu')."
    )
    sam_grp.add_argument(
        "--sam-dtype", type=str, choices=["bfloat16", "float16", "float32"], default=None,
        help="Torch dtype for SAM 3.1."
    )
    sam_grp.add_argument(
        "--sam-threshold", type=float, default=None,
        help="Confidence cutoff for SAM detection."
    )
    sam_grp.add_argument(
        "--mask-dilate", type=int, default=None,
        help="Pixel radius to dilate mask to cover contact shadows."
    )
    sam_grp.add_argument(
        "--sam-prompt", action="append", default=None,
        help="Furniture prompt to segment. Repeatable for multiple prompts."
    )
    sam_grp.add_argument(
        "--sam-skip-prompt", action="append", default=None,
        help="Prompt to segment and save separately, but exclude from final union mask. Repeatable."
    )

    # Stable Diffusion Inpainting Options
    sd_grp = parser.add_argument_group("Stable Diffusion Inpainting Options (--sd-*)")
    sd_grp.add_argument(
        "--sd-model-path", type=str, default=None,
        help="SD Inpainting Hugging Face ID or local path."
    )
    sd_grp.add_argument(
        "--sd-device", type=str, default=None,
        help="Device for Stable Diffusion inference ('cuda' or 'cpu')."
    )
    sd_grp.add_argument(
        "--sd-dtype", type=str, choices=["float16", "bfloat16", "float32"], default=None,
        help="Torch dtype for Stable Diffusion."
    )
    sd_grp.add_argument(
        "--sd-scheduler", type=str, choices=["dpm++", "euler_a", "pndm", "ddim"], default=None,
        help="Diffusion noise scheduler."
    )
    sd_grp.add_argument(
        "--sd-steps", type=int, default=None,
        help="Inference steps for diffusion model."
    )
    sd_grp.add_argument(
        "--sd-guidance", type=float, default=None,
        help="Classifier-Free Guidance (CFG) scale."
    )
    sd_grp.add_argument(
        "--sd-strength", type=float, default=None,
        help="Inpainting denoising strength (0.0 to 1.0)."
    )
    sd_grp.add_argument(
        "--sd-seed", type=int, default=None,
        help="Random seed for generation reproducibility."
    )
    sd_grp.add_argument(
        "--sd-attention-slicing", action="store_true", default=None,
        help="Enable attention slicing for lower VRAM usage."
    )
    sd_grp.add_argument(
        "--sd-prompt", type=str, default=None,
        help="Positive prompt conditioning for empty room."
    )
    sd_grp.add_argument(
        "--sd-negative-prompt", type=str, default=None,
        help="Negative prompt discouraging furniture and artifacts."
    )

    # Geometry & Blending Options
    geom_grp = parser.add_argument_group("Geometry & Blending Options")
    geom_grp.add_argument(
        "--no-roll", action="store_true", default=None,
        help="Disable horizontal circular roll (centroid centering)."
    )
    geom_grp.add_argument(
        "--no-pad", action="store_true", default=None,
        help="Disable horizontal circular wrap padding."
    )
    geom_grp.add_argument(
        "--pad-ratio", type=float, default=None,
        help="Fraction of image width to pad circularly on each side."
    )
    geom_grp.add_argument(
        "--target-height", type=int, default=None,
        help="Inpaint height (preserves aspect ratio, width multiple of 8)."
    )
    geom_grp.add_argument(
        "--blend-feather", type=int, default=None,
        help="Pixel radius for alpha boundary feathering."
    )
    geom_grp.add_argument(
        "--output-format", type=str, choices=["jpg", "png"], default=None,
        help="File format for final saved image."
    )
    geom_grp.add_argument(
        "--jpeg-quality", type=int, default=None,
        help="JPEG quality level (1 to 100)."
    )

    return parser


def load_config_and_args(args_list: Optional[List[str]] = None) -> PipelineConfig:
    """
    Parses CLI arguments, merges with YAML configuration file, and produces
    a validated PipelineConfig object.
    """
    parser = build_arg_parser()
    parsed_args = parser.parse_args(args_list)

    # 1. Start from default dataclass instances
    pipeline_cfg = PipelineConfig()
    sam_cfg = SAMConfig()
    sd_cfg = SDConfig()
    geom_cfg = GeometryConfig()

    # Initial fallback from environment variables
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if env_token:
        pipeline_cfg.hf_key = env_token

    # 2. Check and load YAML file
    config_path = Path(parsed_args.config)
    yaml_data = load_yaml_config(config_path)

    # Merge YAML 'pipeline' section
    if "pipeline" in yaml_data:
        p_data = yaml_data["pipeline"]
        if "hf_key" in p_data and p_data["hf_key"]:
            pipeline_cfg.hf_key = str(p_data["hf_key"])
        if "log_level" in p_data:
            pipeline_cfg.log_level = p_data["log_level"]
        if "stage" in p_data:
            pipeline_cfg.stage = p_data["stage"]
        if "recursive" in p_data:
            pipeline_cfg.recursive = bool(p_data["recursive"])
        if "overwrite" in p_data:
            pipeline_cfg.overwrite = bool(p_data["overwrite"])

    # Merge YAML 'sam' section
    if "sam" in yaml_data:
        s_data = yaml_data["sam"]
        if "model_path" in s_data:
            sam_cfg.model_path = str(s_data["model_path"])
        if "device" in s_data:
            sam_cfg.device = str(s_data["device"])
        if "dtype" in s_data:
            sam_cfg.dtype = str(s_data["dtype"])
        if "threshold" in s_data:
            sam_cfg.threshold = float(s_data["threshold"])
        if "mask_dilate" in s_data:
            sam_cfg.mask_dilate = int(s_data["mask_dilate"])
        if "prompts" in s_data and isinstance(s_data["prompts"], list):
            sam_cfg.prompts = [str(p) for p in s_data["prompts"]]
        if "skip_prompts" in s_data and isinstance(s_data["skip_prompts"], list):
            sam_cfg.skip_prompts = [str(p) for p in s_data["skip_prompts"]]

    # Merge YAML 'sd' section
    if "sd" in yaml_data:
        sd_data = yaml_data["sd"]
        if "model_path" in sd_data:
            sd_cfg.model_path = str(sd_data["model_path"])
        if "device" in sd_data:
            sd_cfg.device = str(sd_data["device"])
        if "dtype" in sd_data:
            sd_cfg.dtype = str(sd_data["dtype"])
        if "scheduler" in sd_data:
            sd_cfg.scheduler = str(sd_data["scheduler"])
        if "steps" in sd_data:
            sd_cfg.steps = int(sd_data["steps"])
        if "guidance_scale" in sd_data:
            sd_cfg.guidance_scale = float(sd_data["guidance_scale"])
        if "strength" in sd_data:
            sd_cfg.strength = float(sd_data["strength"])
        if "seed" in sd_data:
            sd_cfg.seed = int(sd_data["seed"])
        if "attention_slicing" in sd_data:
            sd_cfg.attention_slicing = bool(sd_data["attention_slicing"])
        if "prompt" in sd_data:
            sd_cfg.prompt = str(sd_data["prompt"])
        if "negative_prompt" in sd_data:
            sd_cfg.negative_prompt = str(sd_data["negative_prompt"])

    # Merge YAML 'geometry' section
    if "geometry" in yaml_data:
        g_data = yaml_data["geometry"]
        if "roll" in g_data:
            geom_cfg.roll = bool(g_data["roll"])
        if "pad" in g_data:
            geom_cfg.pad = bool(g_data["pad"])
        if "pad_ratio" in g_data:
            geom_cfg.pad_ratio = float(g_data["pad_ratio"])
        if "target_height" in g_data:
            geom_cfg.target_height = int(g_data["target_height"])
        if "blend_feather" in g_data:
            geom_cfg.blend_feather = int(g_data["blend_feather"])
        if "output_format" in g_data:
            geom_cfg.output_format = str(g_data["output_format"])
        if "jpeg_quality" in g_data:
            geom_cfg.jpeg_quality = int(g_data["jpeg_quality"])

    # 3. Explicit CLI argument overrides
    if parsed_args.input is not None:
        pipeline_cfg.input = parsed_args.input
    if parsed_args.output is not None:
        pipeline_cfg.output = parsed_args.output
    if parsed_args.stage is not None:
        pipeline_cfg.stage = parsed_args.stage
    if parsed_args.log_level is not None:
        pipeline_cfg.log_level = parsed_args.log_level
    if parsed_args.hf_key is not None:
        pipeline_cfg.hf_key = parsed_args.hf_key
    if parsed_args.recursive is not None:
        pipeline_cfg.recursive = parsed_args.recursive
    if parsed_args.overwrite is not None:
        pipeline_cfg.overwrite = parsed_args.overwrite

    # Propagate HF key to environment and submodules
    if pipeline_cfg.hf_key:
        sam_cfg.hf_key = pipeline_cfg.hf_key
        sd_cfg.hf_key = pipeline_cfg.hf_key
        os.environ["HF_TOKEN"] = pipeline_cfg.hf_key
        os.environ["HUGGING_FACE_HUB_TOKEN"] = pipeline_cfg.hf_key

    # SAM overrides
    if parsed_args.sam_model_path is not None:
        sam_cfg.model_path = parsed_args.sam_model_path
    if parsed_args.sam_device is not None:
        sam_cfg.device = parsed_args.sam_device
    if parsed_args.sam_dtype is not None:
        sam_cfg.dtype = parsed_args.sam_dtype
    if parsed_args.sam_threshold is not None:
        sam_cfg.threshold = parsed_args.sam_threshold
    if parsed_args.mask_dilate is not None:
        sam_cfg.mask_dilate = parsed_args.mask_dilate
    if parsed_args.sam_prompt is not None:
        sam_cfg.prompts = parsed_args.sam_prompt
    if parsed_args.sam_skip_prompt is not None:
        sam_cfg.skip_prompts = parsed_args.sam_skip_prompt

    # SD overrides
    if parsed_args.sd_model_path is not None:
        sd_cfg.model_path = parsed_args.sd_model_path
    if parsed_args.sd_device is not None:
        sd_cfg.device = parsed_args.sd_device
    if parsed_args.sd_dtype is not None:
        sd_cfg.dtype = parsed_args.sd_dtype
    if parsed_args.sd_scheduler is not None:
        sd_cfg.scheduler = parsed_args.sd_scheduler
    if parsed_args.sd_steps is not None:
        sd_cfg.steps = parsed_args.sd_steps
    if parsed_args.sd_guidance is not None:
        sd_cfg.guidance_scale = parsed_args.sd_guidance
    if parsed_args.sd_strength is not None:
        sd_cfg.strength = parsed_args.sd_strength
    if parsed_args.sd_seed is not None:
        sd_cfg.seed = parsed_args.sd_seed
    if parsed_args.sd_attention_slicing is not None:
        sd_cfg.attention_slicing = parsed_args.sd_attention_slicing
    if parsed_args.sd_prompt is not None:
        sd_cfg.prompt = parsed_args.sd_prompt
    if parsed_args.sd_negative_prompt is not None:
        sd_cfg.negative_prompt = parsed_args.sd_negative_prompt

    # Geometry overrides
    if parsed_args.no_roll is not None and parsed_args.no_roll:
        geom_cfg.roll = False
    if parsed_args.no_pad is not None and parsed_args.no_pad:
        geom_cfg.pad = False
    if parsed_args.pad_ratio is not None:
        geom_cfg.pad_ratio = parsed_args.pad_ratio
    if parsed_args.target_height is not None:
        geom_cfg.target_height = parsed_args.target_height
    if parsed_args.blend_feather is not None:
        geom_cfg.blend_feather = parsed_args.blend_feather
    if parsed_args.output_format is not None:
        geom_cfg.output_format = parsed_args.output_format
    if parsed_args.jpeg_quality is not None:
        geom_cfg.jpeg_quality = parsed_args.jpeg_quality

    pipeline_cfg.sam = sam_cfg
    pipeline_cfg.sd = sd_cfg
    pipeline_cfg.geometry = geom_cfg

    return pipeline_cfg
