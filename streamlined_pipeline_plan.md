# Matterport Defurnishing Pipeline Replication — Technical Specification

Replicating the pipeline from:
> **"An Empty Room is All We Want: Automatic Defurnishing of Indoor Panoramas"**  
> *Mira Slavcheva, Dave Gausebeck, Kevin Chen, David Buchhofer, Azwad Sabik, Chen Ma, Sachal Dhillon, Olaf Brandt, Alan Dolhasz (CVPRW 2024)*

---

## 1. Pipeline Overview & Flow

```text
[Input: Single 360° Panorama OR Folder of Panoramas]
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 1: SAM 3.1 MASK GENERATION (Loaded ONCE in VRAM)   │
│  - Load prompts & neg-prompts (from config or CLI)       │
│  - Run --sam-refine detection passes (1 to 5, def: 3)    │
│  - Subtract negative prompts (preserve floor/walls)      │
│  - Pixel-wise boolean union across all refine passes     │
│  - Optional morphology: --mask-dilate to cover shadows   │
│  - Save: masks/run_*.png and masks/final_mask.png        │
│  - Unload SAM 3.1 & free VRAM (torch.cuda.empty_cache()) │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2: PRE-PROCESSING & GEOMETRY (CPU / OpenCV)        │
│  - Roll: Shift image & mask by Δx to center furniture    │
│    (Toggleable via --no-roll for A/B testing)            │
│  - Wrap-Pad: Circularly pad left & right edges (~20%)    │
│    (Toggleable via --no-pad for A/B testing)             │
│  - Downsample: Scale to SD inpainting resolution         │
│  - Save: intermediate_outputs/rolled_padded_*.png        │
│          intermediate_outputs/transform_meta.json        │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 3: STABLE DIFFUSION INPAINTING (Loaded ONCE in VRAM│
│  - Load SD Inpainting into VRAM with --sd-dtype          │
│  - Load prompt & negative prompt (from config or CLI)    │
│  - Inpaint prepared (rolled + padded) image & mask       │
│  - Guided by --sd-scheduler, --sd-steps, --sd-guidance   │
│  - Save: intermediate_outputs/inpaint_raw.png            │
│  - Unload SD Inpainter & free VRAM                       │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 4: POST-PROCESSING & BLENDING (CPU / OpenCV)       │
│  - Resolution Restoration: Upsample inpaint to padded res│
│  - Unpad: Crop out the left & right wrap padding         │
│  - Unroll: Shift back by -Δx to original 360° coordinate │
│  - Mask Feathering: Soft transition border (--blend-feather)
│  - High-Frequency Blend:                                 │
│      Final = (Inpainted × Alpha) + (Original × (1-Alpha))│
│  - Save: final_output.jpg (or .png)                      │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Default Configuration File (`config.yaml`)

To allow **"direct hit" execution** without typing dozens of command-line arguments each time, all core prompts, negative prompts, and model settings are centralized in a default configuration file. 

Running simply:
```bash
python run_pipeline.py --input ./panos --output ./output
```
automatically uses these rich defaults. Any CLI flag will override the corresponding config entry.

### `config.yaml` Structure:
```yaml
# ==============================================================================
# Matterport Defurnishing Pipeline Configuration
# ==============================================================================

# --- SAM 3.1 Mask Generator Settings ---
sam:
  model_path: "facebook/sam3.1"
  device: "cuda"
  dtype: "bfloat16"               # bfloat16, float16, or float32
  refine: 3                       # Number of multi-pass refinement runs (1-5)
  threshold: 0.40                 # Detection confidence threshold
  mask_dilate: 10                 # Pixels to expand mask to cover furniture shadows
  prompts:
    - "sofa"
    - "couch"
    - "chair"
    - "armchair"
    - "table"
    - "coffee table"
    - "bed"
    - "nightstand"
    - "desk"
    - "cabinet"
    - "wardrobe"
    - "bookshelf"
    - "tv stand"
    - "dresser"
    - "rug"
    - "carpet"
    - "lamp"
  negative_prompts:
    - "wall"
    - "floor"
    - "ceiling"
    - "window"
    - "door"
    - "column"

# --- Stable Diffusion Inpainting Settings ---
sd:
  model_path: "stable-diffusion-v1-5/stable-diffusion-inpainting"
  device: "cuda"
  dtype: "float16"                # float16, bfloat16, or float32
  scheduler: "dpm++"              # dpm++, euler_a, pndm, ddim
  steps: 25                       # 20-25 steps for dpm++, 50 for pndm
  guidance_scale: 7.5
  strength: 0.99
  seed: 1234
  attention_slicing: false        # Enable for lower VRAM consumption (consumer GPUs)
  prompt: "empty unfurnished room interior, clean bare floor, clear walls, realistic indoor architecture, photorealistic, high resolution"
  negative_prompt: "furniture, sofa, chair, table, bed, desk, cabinet, clutter, shadows, distorted walls, artifacts, text, watermark, blurry"

# --- Geometry & Pre/Post-Processing ---
geometry:
  roll: true                      # Circular horizontal roll to center furniture
  pad: true                       # Circular wrap-padding at 0°/360° seams
  pad_ratio: 0.20                 # Fraction of image width to pad on each side
  target_height: 512              # Inpaint height (aspect ratio preserved, width mult of 8)
  blend_feather: 6                # Pixel radius for boundary alpha feathering
  output_format: "jpg"            # jpg or png
  jpeg_quality: 95
```

---

## 3. Complete CLI Reference Table

Every argument is explicitly prefixed to eliminate any ambiguity between SAM and Stable Diffusion:

### A. SAM 3.1 Options (`--sam-*`)
| Argument | Type | Default (from config) | Description |
|---|---|---|---|
| `--sam-model-path` | String | `facebook/sam3.1` | Local checkpoint path or HuggingFace repo ID |
| `--sam-device` | String | `cuda` | Device (`cuda` or `cpu`) |
| `--sam-dtype` | Choice | `bfloat16` | Precision: `bfloat16`, `float16`, or `float32` |
| `--sam-prompt` | String/List | *From config.yaml* | Furniture concepts to segment (repeatable / multi-arg) |
| `--sam-negative-prompt`| String/List | *From config.yaml* | Structural concepts to exclude (e.g. wall, floor) |
| `--sam-prompt-file` | Path | `None` | Optional text/YAML file containing prompt lists |
| `--sam-refine` | Int (1–5) | `3` | Refinement passes per image to catch missed furniture |
| `--sam-threshold` | Float | `0.40` | Detection confidence cutoff |
| `--mask-dilate` | Int | `10` | Pixels to dilate mask to swallow furniture contact shadows |

### B. Stable Diffusion Inpainting Options (`--sd-*`)
| Argument | Type | Default (from config) | Description |
|---|---|---|---|
| `--sd-model-path` | String | `stable-diffusion-v1-5/stable-diffusion-inpainting` | Local checkpoint path or HuggingFace model ID |
| `--sd-device` | String | `cuda` | Device (`cuda` or `cpu`) |
| `--sd-dtype` | Choice | `float16` | Precision: `float16`, `bfloat16`, or `float32` |
| `--sd-prompt` | String | *From config.yaml* | Positive prompt describing the empty room |
| `--sd-negative-prompt`| String | *From config.yaml* | Negative prompt discouraging furniture/artifacts |
| `--sd-scheduler` | Choice | `dpm++` | Denoising sampler: `dpm++`, `euler_a`, `pndm`, `ddim` |
| `--sd-steps` | Int | `25` | Diffusion inference steps |
| `--sd-guidance` | Float | `7.5` | Classifier-Free Guidance (CFG) scale |
| `--sd-strength` | Float | `0.99` | Inpainting denoising strength (0.0 to 1.0) |
| `--sd-seed` | Int | `1234` | Random seed for generation reproducibility |
| `--sd-attention-slicing`| Flag | `False` | Cuts VRAM for low-memory GPUs (e.g. 6GB–8GB) |

### C. General, Geometry & Output Options
| Argument | Type | Default | Description |
|---|---|---|---|
| `--input` | Path | *Required* | Path to single panorama file or directory of panoramas |
| `--output` | Path | `./output` | Output root directory |
| `--config` | Path | `config.yaml` | Path to custom YAML configuration file |
| `--stage` | Choice | `all` | Execution mode: `all`, `masks-only`, or `inpaint-only` |
| `--no-roll` | Flag | `False` | Disable horizontal roll (for Matterport A/B testing) |
| `--no-pad` | Flag | `False` | Disable wrap padding (for Matterport A/B testing) |
| `--pad-ratio` | Float | `0.20` | Wrap-pad ratio per side (fraction of image width) |
| `--target-height` | Int | `512` | Working inpaint height (aspect ratio kept, mult of 8) |
| `--blend-feather` | Int | `6` | Alpha feather boundary width in pixels |
| `--output-format` | Choice | `jpg` | Final image format: `jpg` or `png` |
| `--jpeg-quality` | Int | `95` | Quality for saved JPEG files (1–100) |
| `--recursive` | Flag | `False` | Recursively scan subfolders when `--input` is a folder |
| `--overwrite` | Flag | `False` | Overwrite existing outputs if already completed |

---

## 4. Detailed Stage Breakdown (Replicating Matterport)

### Stage 1: SAM 3.1 Mask Generation with Multi-Pass `--sam-refine`
1. For each panorama in the batch:
   - Run $N$ refinement passes ($N =$ `--sam-refine`):
     - **Pass 1:** Primary furniture concepts from config.
     - **Pass 2:** Extended secondary concepts (tables, rugs, lamps, sideboards).
     - **Pass 3:** High-sensitivity pass (`--sam-threshold - 0.10`) for distant/occluded items.
   - Subtract any overlapping regions from `--sam-negative-prompt` (e.g. floor, walls).
   - Compute the boolean union across all passes:
     $$\text{Mask}_{\text{union}} = \bigcup_{i=1}^{N} \text{Mask}_{\text{pass}_i}$$
   - Apply morphological dilation with radius `--mask-dilate` (default 10px) to expand the mask over furniture contact shadows.
   - Save intermediate pass masks (`masks/run_1.png`...`masks/run_N.png`) and `masks/final_mask.png`.
2. Unload SAM 3.1 and release GPU memory (`torch.cuda.empty_cache()`).

---

### Stage 2: Equirectangular Pre-processing (Roll & Wrap-Pad)
1. **Centering (Roll):**
   - Calculate horizontal centroid $x_{\text{center}}$ of `final_mask.png`.
   - Circularly shift image and mask horizontally by $\Delta x = \lfloor \frac{W}{2} - x_{\text{center}} \rfloor$.
   - *(Disabled if `--no-roll` is flagged).*
2. **Wrap-Padding:**
   - Circularly duplicate $P = \lfloor W \times \text{pad\_ratio} \rfloor$ pixels from the right onto the left, and left onto the right.
   - *(Disabled if `--no-pad` is flagged).*
3. **Downsampling:**
   - Scale image (Lanczos) and mask (Nearest Neighbor) to `target_height` (e.g. 512).
   - Save `intermediate_outputs/rolled_padded_image.png`, `rolled_padded_mask.png`, and `transform_meta.json`.

---

### Stage 3: Stable Diffusion Inpainting
1. Load SD Inpainting once into VRAM using `--sd-dtype` (FP16 default).
2. For each prepared panorama in the batch:
   - Inpaint using `--sd-prompt`, `--sd-negative-prompt`, `--sd-scheduler` (`dpm++`), and `--sd-steps` (25).
   - Save raw output to `intermediate_outputs/inpaint_raw.png`.
3. Unload SD Inpainting and release GPU memory (`torch.cuda.empty_cache()`).

---

### Stage 4: Post-processing & High-Frequency Blending
1. **Resolution Restoration:** Scale `inpaint_raw.png` back to high-resolution padded dimensions $(W + 2P, H)$.
2. **Unpad & Unroll:** Crop out the padded margins and reverse the horizontal shift by $-\Delta x$.
3. **High-Frequency Composite:**
   - Create a soft feather border on `final_mask.png` with radius `--blend-feather`.
   - Alpha blend against the original high-resolution panorama:
     $$\text{Final} = (\text{Inpainted} \times \text{Alpha}) + (\text{Original} \times (1 - \text{Alpha}))$$
   - All untouched camera pixels (walls, ceiling, untouched floor) remain 100% original.
   - Save `final_output.jpg`.

---

## 5. Output Directory Structure

Matches `pipeline.txt` exactly:

### Case A: Single File Input (`--input ./input/room1.png`)
```text
output/
└── room1/
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

### Case B: Directory Input (`--input ./input/panos/`)
```text
output/
└── panos/
    ├── pano_01/
    │   ├── masks/
    │   │   ├── run_1.png
    │   │   ├── run_2.png
    │   │   ├── run_3.png
    │   │   └── final_mask.png
    │   ├── intermediate_outputs/
    │   │   ├── rolled_padded_image.png
    │   │   ├── rolled_padded_mask.png
    │   │   ├── inpaint_raw.png
    │   │   └── transform_meta.json
    │   └── final_output.jpg
    └── pano_02/
        ├── masks/
        ├── intermediate_outputs/
        └── final_output.jpg
```

---

## 6. Execution Examples

### 1. Direct Hit (Uses all pre-configured defaults from `config.yaml`):
```powershell
python run_pipeline.py --input "./panos" --output "./output"
```

### 2. Custom Prompts & High Refinement:
```powershell
python run_pipeline.py `
  --input "./panos" `
  --output "./output" `
  --sam-refine 4 `
  --mask-dilate 12 `
  --sd-steps 30 `
  --sd-guidance 8.0
```

### 3. Low-VRAM / Consumer GPU Run:
```powershell
python run_pipeline.py `
  --input "./panos" `
  --output "./output" `
  --sam-dtype float16 `
  --sd-dtype float16 `
  --sd-attention-slicing
```

### 4. Matterport Seam A/B Comparison Run:
```powershell
# Run with roll and pad enabled (Matterport replication)
python run_pipeline.py --input "./panos/test.jpg" --output "./output/with_roll_pad"

# Run with roll and pad disabled (baseline comparison)
python run_pipeline.py --input "./panos/test.jpg" --output "./output/without_roll_pad" --no-roll --no-pad
```
