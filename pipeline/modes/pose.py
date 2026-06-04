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
