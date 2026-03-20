import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image

_pipe = None

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

def generate_images(schema: dict, edge_map: Image.Image, n=3) -> list:
    pipe = _load_pipe()
    positive, negative = build_prompt(schema)
    edge_map = edge_map.resize((512, 512))
    images = []
    for seed in range(n):
        gen = torch.Generator().manual_seed(seed * 42)
        out = pipe(
            positive,
            image=edge_map,
            negative_prompt=negative,
            num_inference_steps=25,
            controlnet_conditioning_scale=0.6,
            generator=gen
        ).images[0]
        images.append(out)
    return images