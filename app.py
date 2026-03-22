"""
app.py — Gradio UI (v1) for Vibe-to-Space
Day 2–3 deliverable: text input, optional sketch upload, top-3 gallery, regenerate button.
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── local module imports (same package) ──────────────────────────────────────
from pipeline.parser import parse_prompt
from pipeline.layout import make_edge_map
from pipeline.generate import generate_images
from pipeline.rank import rank_images


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def sketch_to_edge_map(sketch_img: np.ndarray | None) -> Image.Image | None:
    """
    Convert a user-uploaded sketch (numpy RGBA/RGB array from Gradio) into
    a Canny edge map PIL image.  Returns None if no sketch was provided.
    """
    if sketch_img is None:
        return None

    # Gradio returns numpy arrays; convert to PIL then to grayscale
    pil = Image.fromarray(sketch_img).convert("L")
    gray = np.array(pil)

    # Mild blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # Morphological closing to consolidate fragmented edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(edges).convert("RGB")


def run_pipeline(
    vibe_text: str,
    sketch_img: np.ndarray | None,
    seed_offset: int,
) -> tuple[list[Image.Image], str]:
    """
    Full text → image pipeline.
    Returns (list_of_3_pil_images, status_message).
    """
    if not vibe_text.strip():
        return [], "⚠️  Please enter a vibe description."

    # 1. LLM parse ─────────────────────────────────────────────────────────
    try:
        schema = parse_prompt(vibe_text)
        schema_str = str(schema)
    except Exception as e:
        return [], f"❌  LLM parsing failed: {e}"

    # 2. Edge map ──────────────────────────────────────────────────────────
    edge_map = sketch_to_edge_map(sketch_img)
    if edge_map is None:
        # Fall back to procedural layout from parsed room list
        rooms = schema.get("rooms", ["living room"])
        edge_map = make_edge_map(rooms)

    # 3. Generate candidate images ─────────────────────────────────────────
    try:
        candidates = generate_images(schema, edge_map, n=6, seed_offset=seed_offset)
    except Exception as e:
        return [], f"❌  Image generation failed: {e}"

    # 4. CLIP rank → top 3 ────────────────────────────────────────────────
    try:
        top3 = rank_images(candidates, vibe_text)[:3]
    except Exception as e:
        top3 = candidates[:3]   # graceful fallback if CLIP fails

    status = f"✅  Done!  Parsed schema: {schema_str}"
    return top3, status


# ─────────────────────────────────────────────────────────────────────────────
# State helper — tracks a seed offset so "Regenerate" produces new images
# ─────────────────────────────────────────────────────────────────────────────

def generate_handler(vibe_text, sketch_img, seed_state):
    images, status = run_pipeline(vibe_text, sketch_img, seed_offset=seed_state)
    new_seed = seed_state + 100          # shift seeds for next regeneration
    return images, status, new_seed


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI layout
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Vibe-to-Space",
    theme=gr.themes.Soft(primary_hue="slate"),
    css="""
        #header { text-align: center; margin-bottom: 8px; }
        #gallery { min-height: 340px; }
        #status_box { font-size: 0.82em; color: #555; }
    """,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        # 🏠 Vibe-to-Space
        ### A Human–AI Co-Creative Interior Design System
        Describe the *feeling* of a space. Upload an optional sketch.
        We'll generate three interior design renders that match your vibe.
        """,
        elem_id="header",
    )

    # ── Hidden state: seed offset (integer) ──────────────────────────────
    seed_state = gr.State(value=0)

    # ── Main row ─────────────────────────────────────────────────────────
    with gr.Row():

        # Left column — inputs
        with gr.Column(scale=1, min_width=300):
            vibe_input = gr.Textbox(
                label="Describe your vibe",
                placeholder=(
                    "e.g. A warm Scandinavian living room with oak floors, "
                    "large south-facing windows, late afternoon light, and "
                    "a few potted plants…"
                ),
                lines=5,
            )

            sketch_input = gr.Image(
                label="Upload a sketch (optional)",
                type="numpy",
                sources=["upload", "clipboard"],
                image_mode="RGB",
                height=220,
            )

            with gr.Row():
                generate_btn = gr.Button("✨  Generate", variant="primary")
                regenerate_btn = gr.Button("🔄  Regenerate")

            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3,
                elem_id="status_box",
            )

        # Right column — outputs
        with gr.Column(scale=2, min_width=480):
            gallery = gr.Gallery(
                label="Top-3 Ranked Renders",
                columns=3,
                rows=1,
                height=380,
                object_fit="cover",
                elem_id="gallery",
                show_label=True,
            )

    # ── Accordion: how it works ───────────────────────────────────────────
    with gr.Accordion("ℹ️  How it works", open=False):
        gr.Markdown(
            """
            1. **LLM Parse** — Claude extracts rooms, style, materials, lighting,
               colour palette and negative attributes from your description.
            2. **Edge Map** — If you upload a sketch it is converted to a Canny
               edge map; otherwise a procedural layout is generated from the
               parsed room list.
            3. **ControlNet + SD 1.5** — Six candidate images are generated with
               the edge map as structural conditioning and your parsed attributes
               as the text prompt.
            4. **CLIP Ranking** — All six candidates are scored against your
               original vibe description; the top three are shown.
            5. **Regenerate** shifts the random seeds so you get a fresh batch
               without changing your inputs.
            """
        )

    # ── Examples ─────────────────────────────────────────────────────────
    gr.Examples(
        examples=[
            ["A brutalist concrete loft with exposed ceilings, moody evening light, and a single low sofa.", None],
            ["Japandi bedroom — white linen, bamboo accents, soft morning light filtering through shoji screens.", None],
            ["A cosy mid-century reading nook with warm amber lamp light, walnut shelves, and a worn leather armchair.", None],
        ],
        inputs=[vibe_input, sketch_input],
        label="Example Prompts",
    )

    # ── Event wiring ──────────────────────────────────────────────────────
    # Both buttons call the same handler; regenerate just uses an incremented seed.

    generate_btn.click(
        fn=generate_handler,
        inputs=[vibe_input, sketch_input, seed_state],
        outputs=[gallery, status_box, seed_state],
    )

    regenerate_btn.click(
        fn=generate_handler,
        inputs=[vibe_input, sketch_input, seed_state],
        outputs=[gallery, status_box, seed_state],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # set True to get a public tunnel link
        show_error=True,
    )

