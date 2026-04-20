"""
app.py — Gradio UI (v1) for Vibe-to-Space
Day 2–3 deliverable: text input, top-3 gallery, regenerate button.
"""

import gradio as gr
import numpy as np
from PIL import Image
import sys
import os
import time
from functools import lru_cache
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── local module imports (same package) ──────────────────────────────────────
if os.environ.get("MOCK"):
    from pipeline.mock import (
        parse_prompt,
        make_edge_map,
        generate_images,
        rank_images,
        critique_image,
        render_blueprint,
    )
else:
    from pipeline.parser import parse_prompt
    from pipeline.layout import make_edge_map
    from pipeline.generate import generate_images
    from pipeline.rank import rank_images
    from pipeline.critique import critique_image
    from pipeline.blueprint import render_blueprint


# Skip the LLM call when the same vibe text is reused (e.g. Regenerate clicks).
@lru_cache(maxsize=32)
def cached_parse(vibe_text: str) -> dict:
    return parse_prompt(vibe_text)


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_stream(
    vibe_text: str,
    seed_offset: int,
):
    """
    Generator that runs the two-step pipeline and yields intermediate results.

    Yields tuples of (ranked_images, blueprint, status, schema):
      * first yield:  renders empty, blueprint ready (Step 1 done)
      * second yield: renders ready, blueprint unchanged (Step 2 done)
    """
    if not vibe_text.strip():
        yield [], None, "Please enter a vibe description.", {}
        return

    t0 = time.perf_counter()

    # Step 1a — LLM parse
    try:
        schema = cached_parse(vibe_text)
    except Exception as e:
        yield [], None, f"LLM parsing failed: {e}", {}
        return
    t_parse = time.perf_counter()

    # Step 1b — Blueprint
    try:
        blueprint = render_blueprint(schema)
    except Exception as e:
        print(f"[blueprint] render failed: {e}")
        blueprint = None
    t_bp = time.perf_counter()

    yield (
        [],
        blueprint,
        (
            f"Step 1 complete: blueprint ready "
            f"(parse {t_parse - t0:.1f}s, blueprint {t_bp - t_parse:.1f}s). "
            f"Step 2: generating render..."
        ),
        schema,
    )

    # Step 2a — edge map
    rooms = schema.get("rooms", ["living room"])
    edge_map = make_edge_map(rooms)
    t_edge = time.perf_counter()

    # Step 2b — generate
    try:
        candidates = generate_images(schema, edge_map, n=1, seed_offset=seed_offset)
    except Exception as e:
        yield [], blueprint, f"Image generation failed: {e}", schema
        return
    t_gen = time.perf_counter()

    # Step 2c — rank
    try:
        ranked = rank_images(candidates, vibe_text)
    except Exception:
        ranked = candidates
    t_rank = time.perf_counter()

    timings = (
        f"parse={t_parse - t0:.1f}s "
        f"bp={t_bp - t_parse:.1f}s "
        f"edge={t_edge - t_bp:.1f}s "
        f"gen={t_gen - t_edge:.1f}s "
        f"rank={t_rank - t_gen:.1f}s "
        f"total={t_rank - t0:.1f}s"
    )
    print(f"[run_pipeline_stream] {timings}")
    yield ranked, blueprint, f"Done ({timings}).", schema


# ─────────────────────────────────────────────────────────────────────────────
# State helper — tracks a seed offset so "Regenerate" produces new images
# ─────────────────────────────────────────────────────────────────────────────

def generate_handler(vibe_text, seed_state, history):
    history = [vibe_text]
    new_seed = seed_state + 100
    for ranked, blueprint, status, schema in run_pipeline_stream(
        vibe_text, seed_offset=seed_state
    ):
        yield ranked, blueprint, status, new_seed, history, ranked, schema


def refine_handler(vibe_text, refinement_text, seed_state, history):
    if not refinement_text.strip():
        yield [], None, "Please enter a refinement.", seed_state, history, [], {}
        return
    history = history + [refinement_text]
    full_context = " | ".join(history)
    new_seed = seed_state + 100
    for ranked, blueprint, status, schema in run_pipeline_stream(
        full_context, seed_offset=seed_state
    ):
        yield ranked, blueprint, status, new_seed, history, ranked, schema


def on_gallery_select(evt: gr.SelectData) -> int:
    return evt.index


def critique_handler(images, selected_idx, vibe_text, schema):
    if not images:
        return "Generate some images first."
    if selected_idx is None:
        return "Click an image in the gallery to select it first."
    if selected_idx >= len(images):
        return "Selection is out of range. Regenerate and try again."

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
                "Could not reach Ollama at localhost:11434.\n"
                "Run `ollama serve` and `ollama pull llava`, then try again."
            )
        if "model" in msg and ("not found" in msg or "404" in msg):
            return "LLaVA model not installed. Run `ollama pull llava`."
        return f"Critique failed: {e}"


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
        # Vibe-to-Space
        ### A Human–AI Co-Creative Interior Design System
        Describe the *feeling* of a space.
        We'll generate an interior design render that matches your vibe.
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

            refinement_input = gr.Textbox(
                label="Refinement prompt",
                placeholder=(
                    "e.g. Make it warmer, add more plants, "
                    "darker wood tones…"
                ),
                lines=2,
            )

            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                refine_btn = gr.Button("Refine", variant="secondary")
                regenerate_btn = gr.Button("Regenerate")

            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3,
                elem_id="status_box",
            )

        # Right column — outputs
        with gr.Column(scale=2, min_width=480):
            blueprint_output = gr.Image(
                label="Step 1: Blueprint (floor plan)",
                height=340,
                interactive=False,
                show_label=True,
            )

            gallery = gr.Gallery(
                label="Step 2: Render (click to select)",
                columns=1,
                rows=1,
                height=340,
                object_fit="cover",
                elem_id="gallery",
                show_label=True,
            )

            critique_btn = gr.Button("Critique selected image (LLaVA)")
            critique_box = gr.Textbox(
                label="LLaVA critique",
                interactive=False,
                lines=5,
                placeholder="Generate images, click one in the gallery, then click Critique.",
            )

    # ── Accordion: how it works ───────────────────────────────────────────
    with gr.Accordion("How it works", open=False):
        gr.Markdown(
            """
            **Two-step pipeline:**

            1. **LLM Parse + Blueprint (Step 1)** — Claude extracts rooms, style,
               materials, lighting, furniture, and rough dimensions from your
               description. A labelled floor plan is drawn from that schema and
               shown immediately.
            2. **Edge Map** — a procedural layout is built from the room list.
            3. **ControlNet + SD 1.5 (Step 2)** — a single render is generated
               with the edge map as structural conditioning.
            4. **CLIP Ranking** — the render is scored against your vibe
               description.
            5. **Regenerate** shifts the random seed for a fresh Step 2 render
               while keeping the same blueprint.
            """
        )

    # ── Examples ─────────────────────────────────────────────────────────
    gr.Examples(
        examples=[
            ["A brutalist concrete loft with exposed ceilings, moody evening light, and a single low sofa."],
            ["Japandi bedroom — white linen, bamboo accents, soft morning light filtering through shoji screens."],
            ["A cosy mid-century reading nook with warm amber lamp light, walnut shelves, and a worn leather armchair."],
        ],
        inputs=[vibe_input],
        label="Example Prompts",
    )

    # ── Event wiring ──────────────────────────────────────────────────────
    # Both buttons call the same handler; regenerate just uses an incremented seed.

    generate_btn.click(
        fn=generate_handler,
        inputs=[vibe_input, seed_state, history_state],
        outputs=[gallery, blueprint_output, status_box, seed_state, history_state, images_state, schema_state],
    )

    refine_btn.click(
        fn=refine_handler,
        inputs=[vibe_input, refinement_input, seed_state, history_state],
        outputs=[gallery, blueprint_output, status_box, seed_state, history_state, images_state, schema_state],
    )

    regenerate_btn.click(
        fn=generate_handler,
        inputs=[vibe_input, seed_state, history_state],
        outputs=[gallery, blueprint_output, status_box, seed_state, history_state, images_state, schema_state],
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

def prewarm():
    """Trigger one-time model loads (SD + ControlNet + CLIP) before the first click."""
    if os.environ.get("MOCK"):
        return
    print("Pre-warming SD + ControlNet + CLIP (this takes a minute on first run)...")
    warm_schema = {
        "style": ["modern"],
        "materials": ["wood"],
        "lighting": {"type": "natural", "time_of_day": "afternoon"},
        "color_palette": ["beige"],
        "negative": [],
    }
    warm_edge = make_edge_map(["living room"])
    warm_imgs = generate_images(warm_schema, warm_edge, n=1)
    rank_images(warm_imgs, "modern living room")
    print("Models ready.")


if __name__ == "__main__":
    prewarm()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # set True to get a public tunnel link
        show_error=True,
    )

