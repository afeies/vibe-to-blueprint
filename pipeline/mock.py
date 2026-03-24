"""Fast stubs for UI iteration. Use with MOCK=1 env var."""
import random
from PIL import Image, ImageDraw, ImageFont


def parse_prompt(user_text: str) -> dict:
    return {
        "rooms": ["living room"],
        "style": ["modern", "minimal"],
        "materials": ["wood", "concrete"],
        "lighting": {"type": "natural", "direction": "south", "time_of_day": "afternoon"},
        "color_palette": ["#F5F0EB", "#8B7355", "#2F4F4F"],
        "camera": {"angle": "eye-level", "focal_length": "35mm"},
        "negative": ["clutter"],
    }


def make_edge_map(rooms: list) -> Image.Image:
    img = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 462, 462], outline="black", width=3)
    draw.line([256, 50, 256, 462], fill="black", width=2)
    return img


def generate_images(schema: dict, edge_map: Image.Image, n=3, **kwargs) -> list:
    images = []
    for i in range(n):
        color = (
            random.randint(100, 220),
            random.randint(100, 220),
            random.randint(100, 220),
        )
        img = Image.new("RGB", (512, 512), color)
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"Mock render #{i+1}", fill="white")
        style = ", ".join(schema.get("style", []))
        draw.text((20, 50), style[:60], fill="white")
        images.append(img)
    return images


def rank_images(images: list, text: str) -> list:
    shuffled = list(images)
    random.shuffle(shuffled)
    return shuffled
