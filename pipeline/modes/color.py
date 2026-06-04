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
