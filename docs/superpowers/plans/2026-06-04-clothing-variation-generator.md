# Clothing Variation Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool that takes a photo of a model wearing a garment and generates pose, background, and color variations using IP-Adapter + ControlNet on SDXL.

**Architecture:** Each CLI mode loads its own pipeline variant (ControlNet for pose, base for color, inpaint for background). IP-Adapter is applied to all modes to preserve garment appearance. SAM segments the background for the background swap mode.

**Tech Stack:** Python 3.10+, `diffusers>=0.25`, `controlnet-aux`, `segment-anything`, `Pillow`, `torch` (CUDA), `huggingface-hub`, `accelerate`

---

## File Map

```
llm_test/
├── generate.py                    # CLI entrypoint — arg parsing, validation, dispatch
├── pipeline/
│   ├── __init__.py
│   ├── loader.py                  # load pipeline variant based on mode
│   ├── preprocessors.py           # OpenPose extraction, SAM background masking
│   └── modes/
│       ├── __init__.py
│       ├── color.py               # color/pattern variation (base SDXL + IP-Adapter)
│       ├── background.py          # background swap (inpaint SDXL + IP-Adapter + SAM mask)
│       └── pose.py                # pose variation (ControlNet SDXL + IP-Adapter)
├── tests/
│   ├── test_cli.py                # unit tests for CLI validation (no GPU needed)
│   └── test_pipeline.py           # integration test — runs all 3 modes on fixture image
├── dataset/                       # put your source photos here
├── outputs/                       # generated images land here (auto-created)
└── requirements.txt
```

---

## Task 1: Project setup

**Files:**
- Create: `requirements.txt`
- Create: `pipeline/__init__.py`
- Create: `pipeline/modes/__init__.py`
- Create: `tests/__init__.py`
- Create: `dataset/.gitkeep`
- Create: `outputs/.gitkeep`

- [ ] **Step 1: Write requirements.txt**

```
torch>=2.0.0
diffusers>=0.25.0
transformers>=4.35.0
accelerate>=0.25.0
controlnet-aux>=0.0.7
segment-anything>=1.0
Pillow>=10.0.0
huggingface-hub>=0.20.0
```

- [ ] **Step 2: Create package init files and placeholder dirs**

```bash
mkdir -p pipeline/modes tests dataset outputs
touch pipeline/__init__.py pipeline/modes/__init__.py tests/__init__.py
touch dataset/.gitkeep outputs/.gitkeep
```

- [ ] **Step 3: Create .gitignore**

```
outputs/
.superpowers/
__pycache__/
*.pyc
*.pth
.env
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt pipeline/__init__.py pipeline/modes/__init__.py tests/__init__.py dataset/.gitkeep outputs/.gitkeep .gitignore
git commit -m "chore: project scaffold"
```

---

## Task 2: Pipeline loader

**Files:**
- Create: `pipeline/loader.py`

- [ ] **Step 1: Write failing test for loader interface**

Create `tests/test_cli.py`:

```python
import pytest
from unittest.mock import patch, MagicMock

def test_load_pipeline_returns_object_for_each_mode():
    """Loader must return a non-None object for every valid mode."""
    with patch("pipeline.loader._load_base_pipeline") as mock_base, \
         patch("pipeline.loader._load_controlnet_pipeline") as mock_ctrl, \
         patch("pipeline.loader._load_inpaint_pipeline") as mock_inpaint:
        mock_base.return_value = MagicMock()
        mock_ctrl.return_value = MagicMock()
        mock_inpaint.return_value = MagicMock()

        from pipeline.loader import load_pipeline
        assert load_pipeline("color") is not None
        assert load_pipeline("pose") is not None
        assert load_pipeline("background") is not None

def test_load_pipeline_raises_on_unknown_mode():
    from pipeline.loader import load_pipeline
    with pytest.raises(ValueError, match="Unknown mode"):
        load_pipeline("invalid")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli.py::test_load_pipeline_returns_object_for_each_mode tests/test_cli.py::test_load_pipeline_raises_on_unknown_mode -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.loader'`

- [ ] **Step 3: Implement pipeline/loader.py**

```python
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
)

SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROLNET_POSE_MODEL = "thibaud/controlnet-openpose-sdxl-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHTS = "ip-adapter_sdxl.bin"
IP_ADAPTER_SCALE = 0.8
DTYPE = torch.float16


def load_pipeline(mode: str):
    """Load the appropriate pipeline for the given mode."""
    if mode == "color":
        return _load_base_pipeline()
    elif mode == "pose":
        return _load_controlnet_pipeline()
    elif mode == "background":
        return _load_inpaint_pipeline()
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose from: color, pose, background")


def _load_base_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL, torch_dtype=DTYPE
    )
    pipe.enable_model_cpu_offload()
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHTS)
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
    return pipe


def _load_controlnet_pipeline():
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_POSE_MODEL, torch_dtype=DTYPE)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_MODEL, controlnet=controlnet, torch_dtype=DTYPE
    )
    pipe.enable_model_cpu_offload()
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHTS)
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
    return pipe


def _load_inpaint_pipeline():
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        SDXL_INPAINT_MODEL, torch_dtype=DTYPE
    )
    pipe.enable_model_cpu_offload()
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHTS)
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
    return pipe
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli.py::test_load_pipeline_returns_object_for_each_mode tests/test_cli.py::test_load_pipeline_raises_on_unknown_mode -v
```

Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/loader.py tests/test_cli.py
git commit -m "feat: add pipeline loader with mode dispatch"
```

---

## Task 3: Preprocessors

**Files:**
- Create: `pipeline/preprocessors.py`

- [ ] **Step 1: Write failing tests for preprocessors**

Add to `tests/test_cli.py`:

```python
from pathlib import Path
from PIL import Image
import numpy as np
import tempfile, os

def _make_dummy_image(path: Path, size=(512, 512)):
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    img.save(path)
    return img

def test_extract_pose_returns_pil_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    _make_dummy_image(img_path)
    with patch("pipeline.preprocessors.OpenposeDetector") as mock_cls:
        mock_detector = MagicMock()
        mock_cls.from_pretrained.return_value = mock_detector
        mock_detector.return_value = Image.new("RGB", (512, 512))
        from pipeline.preprocessors import extract_pose
        result = extract_pose(img_path)
    assert isinstance(result, Image.Image)

def test_segment_background_returns_image_and_mask(tmp_path):
    img_path = tmp_path / "test.jpg"
    _make_dummy_image(img_path)
    with patch("pipeline.preprocessors.sam_model_registry") as mock_reg, \
         patch("pipeline.preprocessors.SamPredictor") as mock_pred_cls:
        mock_sam = MagicMock()
        mock_reg.__getitem__.return_value = MagicMock(return_value=mock_sam)
        mock_predictor = MagicMock()
        mock_pred_cls.return_value = mock_predictor
        mock_predictor.predict.return_value = (
            np.array([np.ones((512, 512), dtype=bool)]),
            np.array([0.95]),
            np.array([0.0]),
        )
        from pipeline.preprocessors import segment_background
        image, mask = segment_background(img_path)
    assert isinstance(image, Image.Image)
    assert isinstance(mask, Image.Image)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py::test_extract_pose_returns_pil_image tests/test_cli.py::test_segment_background_returns_image_and_mask -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.preprocessors'`

- [ ] **Step 3: Implement pipeline/preprocessors.py**

```python
from pathlib import Path
import numpy as np
from PIL import Image
from controlnet_aux import OpenposeDetector
from segment_anything import sam_model_registry, SamPredictor
from huggingface_hub import hf_hub_download


SAM_CHECKPOINT_REPO = "facebook/sam-vit-huge"
SAM_CHECKPOINT_FILE = "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"


def extract_pose(image_path: Path) -> Image.Image:
    """Extract OpenPose skeleton from image. Returns pose map as PIL Image."""
    detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    image = Image.open(image_path).convert("RGB")
    return detector(image)


def segment_background(image_path: Path) -> tuple[Image.Image, Image.Image]:
    """
    Segment the background from an image.
    Uses SAM with a center-point prompt (assumes subject is roughly centered).
    Returns (original_image, background_mask) where mask is white on background.
    """
    import torch

    checkpoint = hf_hub_download(SAM_CHECKPOINT_REPO, SAM_CHECKPOINT_FILE)
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=checkpoint)
    sam.to("cuda")

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    predictor = SamPredictor(sam)
    predictor.set_image(image_np)

    h, w = image_np.shape[:2]
    center_point = np.array([[w // 2, h // 2]])
    center_label = np.array([1])  # foreground point

    masks, scores, _ = predictor.predict(
        point_coords=center_point,
        point_labels=center_label,
        multimask_output=True,
    )
    # Take the highest-confidence mask (the subject/foreground)
    best_mask = masks[np.argmax(scores)]

    # Invert: background is where the subject is NOT
    bg_mask = ~best_mask
    mask_image = Image.fromarray((bg_mask * 255).astype(np.uint8))

    return image, mask_image
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli.py::test_extract_pose_returns_pil_image tests/test_cli.py::test_segment_background_returns_image_and_mask -v
```

Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/preprocessors.py tests/test_cli.py
git commit -m "feat: add pose extraction and background segmentation preprocessors"
```

---

## Task 4: Color variation mode

**Files:**
- Create: `pipeline/modes/color.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_cli.py`:

```python
def test_generate_color_variation_saves_output(tmp_path):
    input_path = tmp_path / "input.jpg"
    output_path = tmp_path / "output.png"
    _make_dummy_image(input_path)

    mock_pipe = MagicMock()
    mock_pipe.return_value.images = [Image.new("RGB", (1024, 1024))]

    from pipeline.modes.color import generate_color_variation
    generate_color_variation(mock_pipe, input_path, "red dress", output_path, resolution=512)

    assert output_path.exists()
    mock_pipe.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli.py::test_generate_color_variation_saves_output -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.modes.color'`

- [ ] **Step 3: Implement pipeline/modes/color.py**

```python
from pathlib import Path
from PIL import Image


def generate_color_variation(pipe, input_path: Path, prompt: str, output_path: Path, resolution: int = 1024) -> None:
    """
    Generate a color/pattern variation of the garment using IP-Adapter + text prompt.
    Lower IP-Adapter scale lets the prompt drive color changes while preserving structure.
    """
    image = Image.open(input_path).convert("RGB").resize((resolution, resolution))

    pipe.set_ip_adapter_scale(0.6)  # lower = more text influence for color change

    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted",
        ip_adapter_image=image,
        num_inference_steps=30,
        guidance_scale=7.5,
        height=resolution,
        width=resolution,
    )
    result.images[0].save(output_path)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_cli.py::test_generate_color_variation_saves_output -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/modes/color.py tests/test_cli.py
git commit -m "feat: add color variation mode"
```

---

## Task 5: Background swap mode

**Files:**
- Create: `pipeline/modes/background.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_cli.py`:

```python
def test_generate_background_variation_saves_output(tmp_path):
    input_path = tmp_path / "input.jpg"
    output_path = tmp_path / "output.png"
    _make_dummy_image(input_path)

    mock_pipe = MagicMock()
    mock_pipe.return_value.images = [Image.new("RGB", (1024, 1024))]
    mock_mask = Image.new("L", (512, 512), color=255)

    with patch("pipeline.modes.background.segment_background") as mock_seg:
        mock_seg.return_value = (Image.new("RGB", (512, 512)), mock_mask)
        from pipeline.modes.background import generate_background_variation
        generate_background_variation(mock_pipe, input_path, "in a forest", output_path, resolution=512)

    assert output_path.exists()
    mock_pipe.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli.py::test_generate_background_variation_saves_output -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.modes.background'`

- [ ] **Step 3: Implement pipeline/modes/background.py**

```python
from pathlib import Path
from PIL import Image
from pipeline.preprocessors import segment_background


def generate_background_variation(pipe, input_path: Path, prompt: str, output_path: Path, resolution: int = 1024) -> None:
    """
    Swap the background by segmenting subject with SAM, then inpainting the background.
    The IP-Adapter reference image ensures the garment appearance is preserved.
    """
    image, bg_mask = segment_background(input_path)
    image = image.resize((resolution, resolution))
    bg_mask = bg_mask.resize((resolution, resolution))

    pipe.set_ip_adapter_scale(0.8)

    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted body",
        image=image,
        mask_image=bg_mask,
        ip_adapter_image=image,
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=0.99,
        height=resolution,
        width=resolution,
    )
    result.images[0].save(output_path)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_cli.py::test_generate_background_variation_saves_output -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/modes/background.py tests/test_cli.py
git commit -m "feat: add background swap mode"
```

---

## Task 6: Pose variation mode

**Files:**
- Create: `pipeline/modes/pose.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_cli.py`:

```python
def test_generate_pose_variation_saves_output(tmp_path):
    input_path = tmp_path / "input.jpg"
    pose_path = tmp_path / "pose.jpg"
    output_path = tmp_path / "output.png"
    _make_dummy_image(input_path)
    _make_dummy_image(pose_path)

    mock_pipe = MagicMock()
    mock_pipe.return_value.images = [Image.new("RGB", (1024, 1024))]

    with patch("pipeline.modes.pose.extract_pose") as mock_pose:
        mock_pose.return_value = Image.new("RGB", (512, 512))
        from pipeline.modes.pose import generate_pose_variation
        generate_pose_variation(mock_pipe, input_path, pose_path, output_path, resolution=512)

    assert output_path.exists()
    mock_pipe.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli.py::test_generate_pose_variation_saves_output -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.modes.pose'`

- [ ] **Step 3: Implement pipeline/modes/pose.py**

```python
from pathlib import Path
from PIL import Image
from pipeline.preprocessors import extract_pose


def generate_pose_variation(pipe, input_path: Path, pose_path: Path, output_path: Path, resolution: int = 1024) -> None:
    """
    Generate a new pose for the model+garment using ControlNet (OpenPose) + IP-Adapter.
    The source image feeds IP-Adapter (garment reference); pose_path drives ControlNet.
    """
    source_image = Image.open(input_path).convert("RGB").resize((resolution, resolution))
    pose_map = extract_pose(pose_path).resize((resolution, resolution))

    pipe.set_ip_adapter_scale(0.8)

    result = pipe(
        prompt="a model wearing clothes, high quality fashion photo, studio lighting",
        negative_prompt="blurry, low quality, distorted limbs, extra limbs",
        image=pose_map,
        ip_adapter_image=source_image,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        height=resolution,
        width=resolution,
    )
    result.images[0].save(output_path)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_cli.py::test_generate_pose_variation_saves_output -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/modes/pose.py tests/test_cli.py
git commit -m "feat: add pose variation mode"
```

---

## Task 7: CLI entrypoint

**Files:**
- Create: `generate.py`

- [ ] **Step 1: Write failing tests for CLI validation**

Add to `tests/test_cli.py`:

```python
import subprocess, sys

def test_cli_missing_input_raises(tmp_path):
    result = subprocess.run(
        [sys.executable, "generate.py", "--input", "nonexistent.jpg", "--mode", "color", "--prompt", "red"],
        capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

def test_cli_unsupported_format_raises(tmp_path):
    bad_file = tmp_path / "image.gif"
    bad_file.write_bytes(b"GIF89a")
    result = subprocess.run(
        [sys.executable, "generate.py", "--input", str(bad_file), "--mode", "color", "--prompt", "red"],
        capture_output=True, text=True
    )
    assert result.returncode != 0

def test_cli_pose_mode_requires_pose_flag(tmp_path):
    img = tmp_path / "img.jpg"
    _make_dummy_image(img)
    result = subprocess.run(
        [sys.executable, "generate.py", "--input", str(img), "--mode", "pose"],
        capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "--pose" in result.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py::test_cli_missing_input_raises tests/test_cli.py::test_cli_unsupported_format_raises tests/test_cli.py::test_cli_pose_mode_requires_pose_flag -v
```

Expected: all FAIL (generate.py doesn't exist yet)

- [ ] **Step 3: Implement generate.py**

```python
import argparse
import sys
from pathlib import Path

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate clothing variations using IP-Adapter + ControlNet")
    parser.add_argument("--input", required=True, type=Path, help="Source image (model wearing garment)")
    parser.add_argument("--mode", required=True, choices=["pose", "background", "color"], help="Variation type")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt (required for background/color modes)")
    parser.add_argument("--pose", type=Path, default=None, help="Reference pose image (required for pose mode)")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory (default: outputs/)")
    parser.add_argument("--resolution", type=int, default=1024, help="Generation resolution (default: 1024, use 768 for low VRAM)")
    return parser.parse_args()


def validate(args):
    if not args.input.exists():
        print(f"Error: input image not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.input.suffix.lower() not in SUPPORTED_FORMATS:
        print(f"Error: unsupported format {args.input.suffix!r}. Use: {', '.join(SUPPORTED_FORMATS)}", file=sys.stderr)
        sys.exit(1)
    if args.mode == "pose" and args.pose is None:
        print("Error: --pose <image> is required for pose mode", file=sys.stderr)
        sys.exit(1)
    if args.mode == "pose" and not args.pose.exists():
        print(f"Error: pose reference image not found: {args.pose}", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()
    validate(args)

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"{args.input.stem}_{args.mode}.png"

    from pipeline.loader import load_pipeline

    try:
        pipe = load_pipeline(args.mode)
    except torch.cuda.OutOfMemoryError:
        print("Error: CUDA out of memory. Try --resolution 768 to reduce VRAM usage.", file=sys.stderr)
        sys.exit(1)

    if args.mode == "color":
        from pipeline.modes.color import generate_color_variation
        generate_color_variation(pipe, args.input, args.prompt, output_path, args.resolution)
    elif args.mode == "background":
        from pipeline.modes.background import generate_background_variation
        generate_background_variation(pipe, args.input, args.prompt, output_path, args.resolution)
    elif args.mode == "pose":
        from pipeline.modes.pose import generate_pose_variation
        generate_pose_variation(pipe, args.input, args.pose, output_path, args.resolution)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import torch
    main()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli.py::test_cli_missing_input_raises tests/test_cli.py::test_cli_unsupported_format_raises tests/test_cli.py::test_cli_pose_mode_requires_pose_flag -v
```

Expected: all PASS

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/test_cli.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add generate.py tests/test_cli.py
git commit -m "feat: add CLI entrypoint with validation"
```

---

## Task 8: Integration test

**Files:**
- Create: `tests/test_pipeline.py`

> **Note:** This test requires a GPU and real model weights (~10GB download on first run). Run manually on Colab/RunPod, not in CI.

- [ ] **Step 1: Add a fixture image to dataset/**

Download any JPEG photo of a person wearing clothing and save it to `dataset/fixture.jpg`.

- [ ] **Step 2: Create tests/test_pipeline.py**

```python
"""
Integration tests — require GPU + downloaded model weights.
Run manually: pytest tests/test_pipeline.py -v -s
"""
import pytest
from pathlib import Path

FIXTURE = Path("dataset/fixture.jpg")
OUTPUT_DIR = Path("outputs/integration_test")


@pytest.fixture(autouse=True)
def check_fixture():
    if not FIXTURE.exists():
        pytest.skip("dataset/fixture.jpg not found — add a photo to run integration tests")


@pytest.fixture(autouse=True)
def output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_color_variation_produces_output():
    from pipeline.loader import load_pipeline
    from pipeline.modes.color import generate_color_variation

    pipe = load_pipeline("color")
    output = OUTPUT_DIR / "color_result.png"
    generate_color_variation(pipe, FIXTURE, "red dress, vibrant color", output, resolution=768)

    assert output.exists()
    assert output.stat().st_size > 10_000  # at least 10KB — not an empty file


def test_background_variation_produces_output():
    from pipeline.loader import load_pipeline
    from pipeline.modes.background import generate_background_variation

    pipe = load_pipeline("background")
    output = OUTPUT_DIR / "background_result.png"
    generate_background_variation(pipe, FIXTURE, "in a bright sunny park", output, resolution=768)

    assert output.exists()
    assert output.stat().st_size > 10_000


def test_pose_variation_produces_output():
    """Uses the fixture itself as the pose reference — result should be similar pose."""
    from pipeline.loader import load_pipeline
    from pipeline.modes.pose import generate_pose_variation

    pipe = load_pipeline("pose")
    output = OUTPUT_DIR / "pose_result.png"
    generate_pose_variation(pipe, FIXTURE, FIXTURE, output, resolution=768)

    assert output.exists()
    assert output.stat().st_size > 10_000
```

- [ ] **Step 3: Run integration tests on GPU machine**

```bash
pytest tests/test_pipeline.py -v -s
```

Expected: all 3 tests PASS, images saved to `outputs/integration_test/`

- [ ] **Step 4: Inspect outputs visually**

Open `outputs/integration_test/color_result.png`, `background_result.png`, `pose_result.png` and verify:
- Garment structure is recognizable
- The requested variation is visible (color changed / background changed / new pose)

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: add integration test for all 3 generation modes"
```

---

## Cloud Setup (Colab/RunPod)

Run these steps once per session before using the tool:

```bash
# Install dependencies (Colab)
!pip install -q torch diffusers transformers accelerate controlnet-aux segment-anything Pillow huggingface-hub

# Clone your repo
!git clone <your-repo-url> llm_test
%cd llm_test

# Run color variation
!python generate.py --input dataset/fixture.jpg --mode color --prompt "blue jacket" --resolution 768
```

For RunPod: use a PyTorch template with CUDA pre-installed, then just `pip install -r requirements.txt`.
