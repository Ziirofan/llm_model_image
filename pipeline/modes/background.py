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
