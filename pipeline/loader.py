SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROLNET_POSE_MODEL = "thibaud/controlnet-openpose-sdxl-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHTS = "ip-adapter_sdxl.bin"
IP_ADAPTER_SCALE = 0.8

try:
    import torch as _torch
    DTYPE = _torch.float16
except ImportError:
    DTYPE = None  # resolved at runtime when torch is available


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
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL, torch_dtype=DTYPE
    )
    pipe.enable_model_cpu_offload()
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHTS)
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
    return pipe


def _load_controlnet_pipeline():
    import torch
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

    controlnet = ControlNetModel.from_pretrained(CONTROLNET_POSE_MODEL, torch_dtype=DTYPE)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_MODEL, controlnet=controlnet, torch_dtype=DTYPE
    )
    pipe.enable_model_cpu_offload()
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHTS)
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
    return pipe


def _load_inpaint_pipeline():
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        SDXL_INPAINT_MODEL, torch_dtype=DTYPE
    )
    pipe.enable_model_cpu_offload()
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHTS)
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
    return pipe
