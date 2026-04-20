import os, json
from dotenv import load_dotenv
import anthropic

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM = """You are a spatial design parser. Convert the user's room description
into a JSON object with exactly these fields:
- rooms: list of room names (primary room first)
- style: list of style descriptors
- materials: list of materials
- lighting: object with keys type, direction, time_of_day
- color_palette: list of colors
- camera: object with keys angle, focal_length
- negative: list of things to avoid
- furniture: list of 3-6 furniture items that belong in the primary room
  (use common nouns like "sofa", "bed", "coffee table", "rug", "lamp",
  "bookshelf", "dining table", "armchair", "plant")
- dimensions: object with keys width_m and length_m giving rough rectangular
  room dimensions in metres. If unspecified, estimate from context
  (living room ~5x4, bedroom ~4x3, studio ~6x5).
Return ONLY valid JSON, no markdown, no explanation."""

def parse_prompt(user_text: str) -> dict:
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_text}]
    )
    raw = msg.content[0].text.strip()
    return json.loads(raw)