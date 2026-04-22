# vibe-to-blueprint
Co-creation system for interior layout and visualization.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your Anthropic API key (required for prompt parsing):

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Or create a `.env` file in the project root with:

```
ANTHROPIC_API_KEY=your_key_here
```

## Running the app

### Gradio UI

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

### CLI

```bash
python main.py
```

### Image generation settings

The Gradio app exposes controls for render size and inference steps on the left side
of the UI. You can change them per generation without restarting the app.

### Mock mode (no API key / GPU required)

Run either entry point with the `MOCK` env var to use stubbed pipeline modules:

```bash
MOCK=1 python app.py
MOCK=1 python main.py
```

## LLaVA critique (optional)

The **"Critique selected image"** button in the Gradio UI sends the chosen render to a
locally-running [Ollama](https://ollama.com) instance with the LLaVA vision model, and
returns a 2–3 sentence analysis of how well the image matches your prompt plus one
concrete suggestion for improvement.

### 1. One-time install

```bash
# install Ollama (macOS — use the installer from ollama.com on Windows/Linux)
brew install ollama

# download the LLaVA model (~4.7 GB, only needed once)
ollama pull llava
```

### 2. Each session — start the Ollama server

Leave this running in its own terminal tab while you use the app:

```bash
ollama serve
```

(Ollama listens on `http://localhost:11434` by default.)

### 3. Using the critique feature in the app

1. Start the Gradio app in another terminal: `python app.py`, then open
   http://localhost:7860.
2. Enter a vibe description and click **Generate** — three ranked renders appear in
   the gallery.
3. **Click one of the renders** in the gallery to select it (it will highlight).
4. Click **🔍 Critique selected image (LLaVA)**.
5. Wait ~10–60 seconds (longer on CPU). The critique appears in the textbox below
   the gallery.

### Configuration (optional)

Override defaults via env vars before launching `app.py`:

```bash
export OLLAMA_URL=http://localhost:11434   # server URL
export LLAVA_MODEL=llava                   # e.g. llava:13b for higher quality
export LLAVA_TIMEOUT=120                   # seconds
```

### Troubleshooting

- **"Could not reach Ollama at localhost:11434"** — the server isn't running.
  Start it with `ollama serve`.
- **"LLaVA model not installed"** — run `ollama pull llava`.
- **Want to try without installing anything?** — `MOCK=1 python app.py` returns a
  canned critique string so you can verify the UI flow end-to-end.
