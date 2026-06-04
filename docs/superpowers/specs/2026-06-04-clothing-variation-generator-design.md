# Clothing Variation Generator — Design Spec

**Date:** 2026-06-04
**Status:** Approved

## Goal

Build a CLI tool that takes a photo of a model wearing a specific garment and generates variations of that image: different poses/angles, different backgrounds/scenes, and color/pattern changes — while preserving the garment's appearance and the model's identity.

## Approach

**IP-Adapter + ControlNet on SDXL** — no fine-tuning required. The source image serves as a visual reference passed through IP-Adapter (garment conditioning), while ControlNet (OpenPose) handles pose control. Background swapping uses inpainting via SAM segmentation. Color variations are driven by text prompts.

Target environment: Google Colab (T4, free tier) or RunPod. ~8GB VRAM sufficient.

## Architecture

```
Source Image + Text Prompt
        ↓
Preprocessors:
  - OpenPose       → skeleton for pose control
  - SAM            → background mask for inpainting
  - IP-Adapter Enc → visual embedding of garment
        ↓
SDXL + IP-Adapter + ControlNet (diffusion core)
        ↓
Generated Variation Image
```

## Project Structure

```
llm_test/
├── generate.py              # CLI entrypoint
├── pipeline/
│   ├── loader.py            # load SDXL + IP-Adapter + ControlNet weights
│   ├── preprocessors.py     # OpenPose extraction, SAM segmentation
│   └── modes/
│       ├── pose.py          # pose variation mode
│       ├── background.py    # background swap mode
│       └── color.py         # color/pattern variation mode
├── dataset/                 # source photos
├── outputs/                 # generated images
├── test_pipeline.py         # integration test
└── requirements.txt
```

## Generation Modes

### Pose variation
Takes a reference pose image (or a target skeleton), runs OpenPose on source, feeds result to ControlNet.

```bash
python generate.py --input dataset/photo.jpg --mode pose --pose dataset/ref_pose.jpg
```

### Background swap
Runs SAM to segment the model+garment from background, inpaints the background with a text prompt.

```bash
python generate.py --input dataset/photo.jpg --mode background --prompt "in a forest"
```

### Color / pattern variation
Passes the source image through IP-Adapter with a modified text prompt describing the new color or pattern. Garment structure is preserved by IP-Adapter conditioning weight.

```bash
python generate.py --input dataset/photo.jpg --mode color --prompt "red dress"
```

## Tech Stack

| Component | Library / Model |
|---|---|
| Diffusion backbone | SDXL (`stabilityai/stable-diffusion-xl-base-1.0`) |
| Garment conditioning | IP-Adapter (`h94/IP-Adapter`, SDXL variant) |
| Pose control | ControlNet OpenPose for SDXL |
| Pose extraction | `controlnet_aux` (OpenPose processor) |
| Background segmentation | `segment-anything` (SAM ViT-H) |
| Orchestration | `diffusers` (HuggingFace) |
| Runtime | Python 3.10+, CUDA |

## Error Handling

- Validate input image path and format (JPEG/PNG) before loading pipeline
- Catch CUDA OOM with a clear message: suggest reducing `--resolution` (default 1024 → 768)
- On batch failure, save successfully generated outputs before raising

## Testing

- `test_pipeline.py`: integration test running all 3 modes on a fixture image, asserts output file is created and non-empty
- Manual visual QA: inspect output images for garment consistency and variation correctness
- No unit tests on diffusion internals (GPU-dependent, too slow for CI)

## Cloud Setup

Recommended: **Google Colab Pro** (A100) or **RunPod** (RTX 3090 / A100 spot).

- Models are downloaded from HuggingFace Hub on first run (~10GB total)
- Use `accelerate` for half-precision (`fp16`) to stay within 8GB VRAM on T4
- Mount Google Drive or use RunPod network volume to persist downloaded weights
