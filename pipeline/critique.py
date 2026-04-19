"""LLaVA-based image critique via a local Ollama server."""
import os
import base64
import io
import requests
from PIL import Image

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLAVA_MODEL = os.getenv("LLAVA_MODEL", "llava")
REQUEST_TIMEOUT = int(os.getenv("LLAVA_TIMEOUT", "120"))


def _encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_prompt(user_text: str, schema: dict) -> str:
    style = ", ".join(schema.get("style", [])) or "n/a"
    materials = ", ".join(schema.get("materials", [])) or "n/a"
    lighting = schema.get("lighting", {}) or {}
    lighting_str = ", ".join(f"{k}: {v}" for k, v in lighting.items()) or "n/a"
    colors = ", ".join(schema.get("color_palette", [])) or "n/a"
    return (
        f'You are an interior design critic. The user asked for:\n"{user_text}"\n\n'
        f"Key attributes parsed from their description:\n"
        f"- Style: {style}\n"
        f"- Materials: {materials}\n"
        f"- Lighting: {lighting_str}\n"
        f"- Colors: {colors}\n\n"
        f"Look at the attached rendered room. In 2-3 sentences:\n"
        f"1) Describe what you see.\n"
        f"2) Say how well it matches the requested vibe.\n"
        f"3) Suggest one concrete change that would improve alignment."
    )


def critique_image(image: Image.Image, user_text: str, schema: dict) -> str:
    prompt = _build_prompt(user_text, schema or {})
    encoded = _encode_image(image)
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLAVA_MODEL,
            "prompt": prompt,
            "images": [encoded],
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()
