# Implementation Plan: Matterport Panorama Defurnishing Pipeline

Replication of the Matterport paper:
> **"An Empty Room is All We Want: Automatic Defurnishing of Indoor Panoramas"** (CVPRW 2024)

Integrating **Meta SAM 3.1** (`dependencies/sam3/`) for refined instance mask generation and **Hugging Face Diffusers** (`dependencies/diffusers/src`) for Stable Diffusion Inpainting with equirectangular 360° geometry handling.

---

## Goal Description
Build a production-grade, highly modular Python pipeline that removes furniture from 360° indoor panoramas:
1. Accepts a single image or a folder of panoramas, strictly following the directory structure and naming conventions defined in `pipeline.txt` / `streamlined_pipeline_plan.md`.
2. Employs a **batch-first model lifecycle**: loads SAM 3.1 once for all images, unloads SAM to free VRAM, then loads Stable Diffusion Inpainting once for all images and unloads it.
3. Implements **`--sam-refine`** (1–5 passes, default 3) to prevent missed furniture, followed by negative prompt subtraction, boolean union, and shadow dilation (`--mask-dilate`).
4. Replicates Matterport's 360° equirectangular geometry pre-processing:
   - **Roll:** Horizontal circular shift to center the primary furniture cluster away from the 0°/360° boundary.
   - **Wrap-Padding:** Circular horizontal padding (e.g. 20%) to provide seamless boundary context.
   - **Downsample:** Scale to SD working dimensions (e.g. 512x256 base, divisible by 8).
5. Inpaints using Stable Diffusion v1.5 with empty-room conditioning.
6. Replicates post-processing: upscales inpaint, unpads, unrolls, feathers the mask boundary, and composites over the original image so that all untouched areas (walls, ceiling, floor) remain 100% original.
7. Supports **"Direct Hit"** execution via `config.yaml` with rich default prompt profiles.
8. External repositories (`diffusers` and `sam3`) are maintained under `dependencies/` and auto-discovered.

---

## User Review Required

> [!IMPORTANT]
> **Hugging Face Access Prerequisite for SAM 3.1:**  
> When running the pipeline on the execution machine, the model weights for `facebook/sam3.1` are gated. The execution machine must have accepted the license terms on Hugging Face and run `huggingface-cli login`.

> [!NOTE]
> **Submodule Auto-Discovery from `dependencies/`:**  
> The pipeline dynamically detects `dependencies/diffusers/src` and `dependencies/sam3` from the project root and injects them into `sys.path` via `src/defurnishing/core/env_setup.py`, allowing seamless execution without requiring global pip installs.

---

## Proposed Changes & Modular Architecture

```text
Impainting Pipeline/
├── config.yaml                            # Master configuration (prompts, thresholds, geometry, SD)
├── requirements.txt                       # Core dependencies (torch, torchvision, opencv-python, pyyaml, etc.)
├── run_pipeline.py                        # Top-level CLI entry point (orchestrator)
│
├── dependencies/                          # Cloned external repositories
│   ├── diffusers/                         # Hugging Face Diffusers repo (src/diffusers)
│   └── sam3/                              # Meta SAM 3.1 repo (sam3/)
│
├── src/                                   # Proper modular source package
│   └── defurnishing/
│       ├── __init__.py
│       │
│       ├── config/                        # Configuration & CLI Parsing
│       │   ├── __init__.py
│       │   ├── schema.py                  # Dataclass schemas for SAM, SD, Geometry & Pipeline configs
│       │   └── loader.py                  # YAML loader and CLI argument parser/merger
│       │
│       ├── core/                          # Utilities & Environment Setup
│       │   ├── __init__.py
│       │   ├── env_setup.py               # Dynamic sys.path resolver for dependencies/
│       │   ├── path_manager.py            # Image discovery, directory hierarchy, output tree builder
│       │   └── logger.py                  # Structured console & file logging
│       │
│       ├── segmentation/                  # Stage 1: SAM 3.1 Mask Generation
│       │   ├── __init__.py
│       │   ├── sam_loader.py              # SAM 3.1 model loader, device & dtype management
│       │   ├── multi_pass.py              # Multi-pass refinement, negative prompt subtraction & union
│       │   └── postprocess.py             # Shadow dilation (morphology) and mask cleanup
│       │
│       ├── geometry/                      # Stage 2: 360° Equirectangular Transformations
│       │   ├── __init__.py
│       │   ├── roll.py                    # Horizontal centroid calculation & circular shift
│       │   ├── padding.py                 # Horizontal wrap-padding for seam prevention
│       │   ├── resize.py                  # High-quality aspect-preserving resize (multiple of 8)
│       │   └── transform.py               # Combined forward (roll+pad+scale) and inverse operations
│       │
│       ├── inpainting/                    # Stage 3: Stable Diffusion Inpainting
│       │   ├── __init__.py
│       │   ├── sd_loader.py               # SD inpainting pipeline loader, schedulers & VRAM optimizations
│       │   └── inpaint_runner.py          # Batch inpainting inference loop
│       │
│       ├── blending/                      # Stage 4: Post-Processing & Compositing
│       │   ├── __init__.py
│       │   ├── feather.py                 # Mask edge feathering (Gaussian / distance transform)
│       │   └── composite.py               # Alpha blending (preserving 100% original camera background)
│       │
│       └── pipeline.py                    # Master pipeline orchestrator coordinating all stages
│
├── tests/                                 # Unit & Integration Tests
│   ├── __init__.py
│   ├── test_geometry.py                   # Pixel-perfect roundtrip tests (roll/unroll, pad/unpad)
│   ├── test_config.py                     # Config parsing and CLI argument overrides test
│   └── test_path_manager.py               # Single-image vs directory discovery tests
│
└── output/                                # Default output directory (generated per run)
    └── <image_or_folder_id>/
        ├── masks/
        │   ├── run_1.png
        │   ├── run_2.png
        │   ├── run_3.png
        │   └── final_mask.png
        ├── intermediate_outputs/
        │   ├── rolled_padded_image.png
        │   ├── rolled_padded_mask.png
        │   ├── inpaint_raw.png
        │   └── transform_meta.json
        └── final_output.jpg
```

---

### Component 1: Configuration & CLI

#### [NEW] `config.yaml`
Stores all default prompts, negative prompts, thresholds, schedulers, and geometry parameters for zero-flag execution (`python run_pipeline.py --input ./panos --output ./output`).

#### [NEW] `src/defurnishing/config/schema.py`
Defines typed configuration dataclasses: `SAMConfig`, `SDConfig`, `GeometryConfig`, `PipelineConfig`.

#### [NEW] `src/defurnishing/config/loader.py`
- Parses `config.yaml`.
- Builds CLI with explicit argument prefixes:
  - `--sam-*` (`model_path`, `device`, `dtype`, `prompt`, `negative_prompt`, `refine`, `threshold`, `mask_dilate`)
  - `--sd-*` (`model_path`, `device`, `dtype`, `prompt`, `negative_prompt`, `scheduler`, `steps`, `guidance`, `strength`, `seed`, `attention_slicing`)
  - Pipeline flags (`--input`, `--output`, `--stage`, `--no-roll`, `--no-pad`, `--pad-ratio`, `--target-height`, `--blend-feather`, `--output-format`, `--jpeg-quality`, `--recursive`, `--overwrite`)
- Merges CLI flags over YAML defaults.

---

### Component 2: Core Utilities & Dependency Bootstrapping

#### [NEW] `src/defurnishing/core/env_setup.py`
- Locates `dependencies/diffusers/src` and `dependencies/sam3` relative to the project root.
- Injects them into `sys.path` dynamically before importing external modules.
- Emits clear diagnostic warnings if submodules are missing or uninitialized.

#### [NEW] `src/defurnishing/core/path_manager.py`
- Distinguishes between single-file and folder inputs.
- Handles recursive discovery and prevents stem collisions.
- Pre-creates folder trees: `masks/`, `intermediate_outputs/`, and `final_output.jpg`.

#### [NEW] `src/defurnishing/core/logger.py`
- Configures structured, human-readable console and optional file logging with stage headers and execution timings.

---

### Component 3: Stage 1 — Refined Mask Generation (Native Meta SAM 3)

#### [NEW] `src/defurnishing/segmentation/sam_loader.py`
- Bootstraps dependencies and imports `build_sam3_image_model` from `dependencies/sam3` and `Sam3Processor` from `sam3.model.sam3_image_processor`.
- Points to the local BPE vocabulary asset: `dependencies/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz`.
- Supports Hugging Face authentication token (`config.hf_key`) for gated model access.
- Loads model with precision `--sam-dtype` (`bfloat16`, `float16`, `float32`).
- Provides explicit VRAM teardown: `del model; del processor; torch.cuda.empty_cache()`.

#### [NEW] `src/defurnishing/segmentation/multi_pass.py`
- Utilizes native `Sam3Processor`:
  - `state = processor.set_image(image_pil)`: visual backbone encodes the panorama **once** and caches the representation.
  - Queries positive furniture concepts with `state = processor.set_text_prompt(prompt, state)`.
  - Queries negative structural concepts (wall, floor, ceiling) with `set_text_prompt`.
  - Resets state prompts between queries via `processor.reset_all_prompts(state)`.
  - Extracts boolean mask arrays directly from `state["masks"]`.
- Runs $N$ refinement passes ($N = 1\dots 5$, default 3) with dynamic threshold adjustments.
- Computes boolean union: $\text{Mask}_{\text{union}} = \bigcup_{i=1}^N \text{Mask}_{\text{pass}_i}$.
- Saves pass masks `masks/run_1.png`...`masks/run_N.png`.

#### [NEW] `src/defurnishing/segmentation/postprocess.py`
- Applies morphological dilation (`--mask-dilate`, default 10px) to expand over furniture contact shadows.
- Saves `masks/final_mask.png`.

---

### Component 4: Stage 2 — Panorama Geometry Pre-processing

#### [NEW] `src/defurnishing/geometry/roll.py`
- Computes horizontal centroid $x_{\text{center}}$ of `final_mask.png`.
- Circularly rolls image and mask by $\Delta x = \lfloor \frac{W}{2} - x_{\text{center}} \rfloor$ (`np.roll`).
- Bypassed if `--no-roll` is specified.

#### [NEW] `src/defurnishing/geometry/padding.py`
- Computes $P = \lfloor W \times \text{pad\_ratio} \rfloor$.
- Circularly wrap-pads left and right edges:
  `np.concatenate([img[:, -P:], img, img[:, :P]], axis=1)`.
- Bypassed if `--no-pad` is specified.

#### [NEW] `src/defurnishing/geometry/resize.py`
- Scales image (Lanczos) and mask (Nearest) to `target_height` (256, default 512x256), ensuring width is divisible by 8.

#### [NEW] `src/defurnishing/geometry/transform.py`
- Chains forward transformations: Roll $\rightarrow$ Wrap-Pad $\rightarrow$ Downsample.
- Emits `intermediate_outputs/rolled_padded_image.png`, `rolled_padded_mask.png`, and `transform_meta.json`.
- Provides inverse transformation: Upsample $\rightarrow$ Unpad $\rightarrow$ Unroll.

---

### Component 5: Stage 3 — Stable Diffusion Inpainting (Native Diffusers)

#### [NEW] `src/defurnishing/inpainting/sd_loader.py`
- Bootstraps dependencies and imports `StableDiffusionInpaintPipeline` and schedulers (`DPMSolverMultistepScheduler`, `EulerAncestralDiscreteScheduler`) directly from `dependencies/diffusers/src`.
- Loads pipeline into VRAM in `--sd-dtype` (`float16`).
- Enables native memory optimizations: `--sd-attention-slicing`, VAE slicing, and VAE tiling.
- Provides explicit VRAM teardown: `del pipe; torch.cuda.empty_cache()`.

#### [NEW] `src/defurnishing/inpainting/inpaint_runner.py`
- Executes batch inpainting with prompt, negative prompt, guidance, steps, and seed.
- Saves raw diffusion output to `intermediate_outputs/inpaint_raw.png`.

---

### Component 6: Stage 4 — Post-Processing & Blending

#### [NEW] `src/defurnishing/blending/feather.py`
- Computes soft alpha feather transition border on `final_mask.png` (radius `--blend-feather`, default 6px).

#### [NEW] `src/defurnishing/blending/composite.py`
- Inverts geometry: Upsamples `inpaint_raw.png` to original padded resolution, unpads margins, and unrolls shift by $-\Delta x$.
- Composites: $\text{Final} = (\text{Inpainted} \times \alpha) + (\text{Original} \times (1 - \alpha))$.
- Untouched non-furniture regions remain 100% original camera pixels.
- Saves `final_output.jpg` with `--jpeg-quality` (or `.png`).

---

### Component 7: Pipeline Orchestrator & CLI Entrypoint

#### [NEW] `src/defurnishing/pipeline.py`
- Manages batch lifecycle:
  1. Stage 1 (SAM 3.1) on all images $\rightarrow$ Unload SAM from GPU.
  2. Stage 2 (CPU Geometry transforms) on all images.
  3. Stage 3 (Stable Diffusion) on all images $\rightarrow$ Unload SD from GPU.
  4. Stage 4 (CPU Blending & Compositing) on all images.
- Supports `--stage all`, `--stage masks-only`, `--stage inpaint-only`.

#### [NEW] `run_pipeline.py`
- Minimal CLI entrypoint invoking `defurnishing.config.loader` and `defurnishing.pipeline.run`.

---

## Verification Plan

### Automated Tests
- **Geometry Roundtrip Test (`tests/test_geometry.py`)**:
  - Test `roll -> unroll` and `wrap_pad -> unpad` on synthetic RGB arrays. Verify 100% pixel-perfect identity.
- **Environment & Dependency Discovery Test**:
  - Verify `setup_dependencies()` discovers `dependencies/diffusers/src` and `dependencies/sam3` and imports succeed.
- **Configuration & CLI Parser Test (`tests/test_config.py`)**:
  - Validate `config.yaml` loading and CLI flag override behavior.
- **Path Manager Test (`tests/test_path_manager.py`)**:
  - Verify single-file vs folder input parsing and output hierarchy generation.

### Manual Verification
- Run `python run_pipeline.py --help` to verify all parameters and formatted help documentation.
- Execute dry-run pipeline test without model inference to verify directory creation, logging, and error handling.
