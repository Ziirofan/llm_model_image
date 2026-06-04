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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)

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
