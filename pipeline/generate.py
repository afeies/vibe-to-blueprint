import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image

_pipe = None
DEFAULT_IMAGE_SIZE = 512
DEFAULT_INFERENCE_STEPS = 25


def _resolve_image_size(image_size: int | None) -> int:
    size = DEFAULT_IMAGE_SIZE if image_size is None else int(image_size)
    if size < 64:
        raise ValueError("image_size must be at least 64 pixels")
    if size % 8 != 0:
        raise ValueError("image_size must be a multiple of 8 pixels")
    return size


def _resolve_inference_steps(num_inference_steps: int | None) -> int:
    steps = DEFAULT_INFERENCE_STEPS if num_inference_steps is None else int(num_inference_steps)
    if steps < 1:
        raise ValueError("num_inference_steps must be at least 1")
    return steps


def _load_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    print("Loading ControlNet + SD1.5 (first run is slow)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    _pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    _pipe.enable_model_cpu_offload()
    return _pipe

def build_prompt(schema: dict) -> tuple[str, str]:
    parts = []
    parts += schema.get("style", [])
    parts += ["interior design", "room"]
    parts += schema.get("materials", [])
    parts += [schema.get("lighting", {}).get("time_of_day", "")]
    parts += schema.get("color_palette", [])
    parts.append("photorealistic, high quality, 8k")
    positive = ", ".join(p for p in parts if p)
    negative = ", ".join(schema.get("negative", [])) + \
               ", cartoon, distorted, low quality, watermark"
    return positive, negative

def generate_images(
    schema: dict,
    edge_map: Image.Image,
    n=3,
    seed_offset=0,
    image_size: int | None = None,
    num_inference_steps: int | None = None,
) -> list:
    pipe = _load_pipe()
    positive, negative = build_prompt(schema)
    image_size = _resolve_image_size(image_size)
    num_inference_steps = _resolve_inference_steps(num_inference_steps)
    target_size = (image_size, image_size)
    if edge_map.size != target_size:
        edge_map = edge_map.resize(target_size, resample=Image.Resampling.NEAREST)
    images = []
    for seed in range(n):
        gen = torch.Generator().manual_seed((seed + seed_offset) * 42)
        out = pipe(
            positive,
            image=edge_map,
            negative_prompt=negative,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=0.6,
            generator=gen
        ).images[0]
        images.append(out)
    return images