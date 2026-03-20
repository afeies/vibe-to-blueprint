import os, json
from dotenv import load_dotenv
import anthropic

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM = """You are a spatial design parser. Convert the user's room description
into a JSON object with exactly these fields:
- rooms: list of room names
- style: list of style descriptors  
- materials: list of materials
- lighting: object with keys type, direction, time_of_day
- color_palette: list of colors
- camera: object with keys angle, focal_length
- negative: list of things to avoid
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