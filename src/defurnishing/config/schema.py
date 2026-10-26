"""
Strongly-typed dataclasses for pipeline configuration.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SAMConfig:
    model_path: str = "facebook/sam3.1"
    device: str = "cuda"
    dtype: str = "bfloat16"
    refine: int = 3
    threshold: float = 0.40
    mask_dilate: int = 10
    hf_key: str = ""
    prompts: List[str] = field(default_factory=lambda: [
        "sofa", "couch", "chair", "armchair", "table", "coffee table",
        "bed", "nightstand", "desk", "cabinet", "wardrobe", "bookshelf",
        "tv stand", "dresser", "rug", "carpet", "lamp"
    ])
    negative_prompts: List[str] = field(default_factory=lambda: [
        "wall", "floor", "ceiling", "window", "door", "column"
    ])


@dataclass
class SDConfig:
    model_path: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    device: str = "cuda"
    dtype: str = "float16"
    scheduler: str = "dpm++"
    steps: int = 25
    guidance_scale: float = 7.5
    strength: float = 0.99
    seed: int = 1234
    attention_slicing: bool = False
    hf_key: str = ""
    prompt: str = (
        "empty unfurnished room interior, clean bare floor, clear walls, "
        "realistic indoor architecture, photorealistic, high resolution"
    )
    negative_prompt: str = (
        "furniture, sofa, chair, table, bed, desk, cabinet, clutter, "
        "shadows, distorted walls, artifacts, text, watermark, blurry"
    )


@dataclass
class GeometryConfig:
    roll: bool = True
    pad: bool = True
    pad_ratio: float = 0.20
    target_height: int = 256
    blend_feather: int = 6
    output_format: str = "jpg"
    jpeg_quality: int = 95


@dataclass
class PipelineConfig:
    input: str = ""
    output: str = "./output"
    config_path: str = "config.yaml"
    stage: str = "all"            # 'all', 'masks-only', 'inpaint-only'
    log_level: str = "INFO"       # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    hf_key: str = ""              # Hugging Face API token / key
    recursive: bool = False
    overwrite: bool = False
    sam: SAMConfig = field(default_factory=SAMConfig)
    sd: SDConfig = field(default_factory=SDConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
