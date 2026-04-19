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
if os.environ.get("MOCK"):
    from pipeline.mock import (
        parse_prompt,
        make_edge_map,
        generate_images,
        rank_images,
        critique_image,
    )
else:
    from pipeline.parser import parse_prompt
    from pipeline.layout import make_edge_map
    from pipeline.generate import generate_images
    from pipeline.rank import rank_images
    from pipeline.critique import critique_image


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
) -> tuple[list[Image.Image], str, dict]:
    """
    Full text → image pipeline.
    Returns (list_of_3_pil_images, status_message, parsed_schema).
    """
    if not vibe_text.strip():
        return [], "⚠️  Please enter a vibe description.", {}

    # 1. LLM parse ─────────────────────────────────────────────────────────
    try:
        schema = parse_prompt(vibe_text)
        schema_str = str(schema)
    except Exception as e:
        return [], f"❌  LLM parsing failed: {e}", {}

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
        return [], f"❌  Image generation failed: {e}", schema

    # 4. CLIP rank → top 3 ────────────────────────────────────────────────
    try:
        top3 = rank_images(candidates, vibe_text)[:3]
    except Exception as e:
        top3 = candidates[:3]   # graceful fallback if CLIP fails

    status = f"✅  Done!  Parsed schema: {schema_str}"
    return top3, status, schema


# ─────────────────────────────────────────────────────────────────────────────
# State helper — tracks a seed offset so "Regenerate" produces new images
# ─────────────────────────────────────────────────────────────────────────────

def generate_handler(vibe_text, sketch_img, seed_state, history):
    history = [vibe_text]
    images, status, schema = run_pipeline(vibe_text, sketch_img, seed_offset=seed_state)
    return images, status, seed_state + 100, history, images, schema


def refine_handler(vibe_text, refinement_text, sketch_img, seed_state, history):
    if not refinement_text.strip():
        return [], "⚠️  Please enter a refinement.", seed_state, history, [], {}
    history = history + [refinement_text]
    full_context = " | ".join(history)
    images, status, schema = run_pipeline(full_context, sketch_img, seed_offset=seed_state)
    return images, status, seed_state + 100, history, images, schema


def on_gallery_select(evt: gr.SelectData) -> int:
    return evt.index


def critique_handler(images, selected_idx, vibe_text, schema):
    if not images:
        return "⚠️  Generate some images first."
    if selected_idx is None:
        return "⚠️  Click an image in the gallery to select it first."
    if selected_idx >= len(images):
        return "⚠️  Selection is out of range — regenerate and try again."

    img = images[selected_idx]
    # Gradio gallery values can come back as PIL, numpy, file path, or (path, caption)
    if isinstance(img, tuple):
        img = img[0]
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    try:
        return critique_image(img, vibe_text, schema or {})
    except Exception as e:
        msg = str(e).lower()
        if "connection" in msg or "refused" in msg:
            return (
                "⚠️  Could not reach Ollama at localhost:11434.\n"
                "Run `ollama serve` and `ollama pull llava`, then try again."
            )
        if "model" in msg and ("not found" in msg or "404" in msg):
            return "⚠️  LLaVA model not installed. Run `ollama pull llava`."
        return f"❌  Critique failed: {e}"


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

    # ── Hidden state: seed, history, current images, parsed schema, selection ──
    seed_state = gr.State(value=0)
    history_state = gr.State(value=[])
    images_state = gr.State(value=[])
    schema_state = gr.State(value={})
    selected_idx_state = gr.State(value=None)

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

            refinement_input = gr.Textbox(
                label="Refinement prompt",
                placeholder=(
                    "e.g. Make it warmer, add more plants, "
                    "darker wood tones…"
                ),
                lines=2,
            )

            with gr.Row():
                generate_btn = gr.Button("✨  Generate", variant="primary")
                refine_btn = gr.Button("🔧  Refine", variant="secondary")
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
                label="Top-3 Ranked Renders (click one to select)",
                columns=3,
                rows=1,
                height=380,
                object_fit="cover",
                elem_id="gallery",
                show_label=True,
            )

            critique_btn = gr.Button("🔍  Critique selected image (LLaVA)")
            critique_box = gr.Textbox(
                label="LLaVA critique",
                interactive=False,
                lines=5,
                placeholder="Generate images, click one in the gallery, then click Critique.",
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
        inputs=[vibe_input, sketch_input, seed_state, history_state],
        outputs=[gallery, status_box, seed_state, history_state, images_state, schema_state],
    )

    refine_btn.click(
        fn=refine_handler,
        inputs=[vibe_input, refinement_input, sketch_input, seed_state, history_state],
        outputs=[gallery, status_box, seed_state, history_state, images_state, schema_state],
    )

    regenerate_btn.click(
        fn=generate_handler,
        inputs=[vibe_input, sketch_input, seed_state, history_state],
        outputs=[gallery, status_box, seed_state, history_state, images_state, schema_state],
    )

    gallery.select(
        fn=on_gallery_select,
        inputs=None,
        outputs=[selected_idx_state],
    )

    critique_btn.click(
        fn=critique_handler,
        inputs=[images_state, selected_idx_state, vibe_input, schema_state],
        outputs=[critique_box],
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

