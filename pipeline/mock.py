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
        "furniture": ["sofa", "coffee table", "bookshelf", "rug", "floor lamp"],
        "dimensions": {"width_m": 5.0, "length_m": 4.0},
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


def render_blueprint(schema: dict) -> Image.Image:
    dims = schema.get("dimensions") or {}
    w = float(dims.get("width_m") or 5.0)
    h = float(dims.get("length_m") or 4.0)
    img = Image.new("RGB", (512, 512), "#FAFAF7")
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 472, 472], outline="#2C2C2C", width=6)
    rooms = ", ".join(schema.get("rooms") or ["room"])
    furn = schema.get("furniture") or []
    draw.text((60, 60), f"[MOCK] Blueprint - {rooms}", fill="#2C2C2C")
    draw.text((60, 90), f"{w:.1f} m x {h:.1f} m", fill="#666")
    for i, item in enumerate(furn[:6]):
        y = 130 + i * 28
        draw.rectangle([60, y, 220, y + 22], fill="#8B7355", outline="#2C2C2C")
        draw.text((70, y + 5), item[:20], fill="white")
    return img


def critique_image(image, user_text: str, schema: dict) -> str:
    style = ", ".join(schema.get("style", [])) or "modern"
    materials = ", ".join(schema.get("materials", [])) or "wood and fabric"
    return (
        f"[MOCK critique] The render shows a {style} interior featuring {materials}, "
        f"with balanced lighting and a clear focal point. It broadly matches the vibe "
        f"you described, though the mood feels slightly cooler than intended. "
        f"Consider warming the palette or adding textured textiles to strengthen the feel."
    )
