# Pitch Visualizer

Pitch Visualizer is a FastAPI web app that turns a pasted narrative into a visual storyboard. It splits the text into scenes with NLTK, optionally refines the image prompts with Anthropic, and generates storyboard panels using Hugging Face image generation.

## What it does

- Accepts a block of story text from a web form.
- Breaks the text into 3 to 5 scenes using NLTK sentence tokenization.
- Optionally refines each scene prompt with Anthropic for stronger visual detail.
- Generates storyboard images with Hugging Face.
- Renders a clean HTML storyboard with Jinja2.
- Lets the user choose a visual style such as Digital Art, Photorealistic, Anime, Watercolor, or Comic.

## Tech Stack

- Python
- FastAPI
- Jinja2
- NLTK
- Anthropic API for optional prompt refinement
- Hugging Face inference API for image generation
- HTML + Tailwind CSS

## Project Structure

- `main.py` - FastAPI app and core logic
- `templates/index.html` - input page
- `templates/storyboard.html` - storyboard output page
- `requirements.txt` - Python dependencies
- `.env` - API keys and model settings

## Setup

### 1. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root with your keys and model settings:

```env
HUGGINGFACE_API_KEY=your_hugging_face_token
HF_IMAGE_MODEL=black-forest-labs/FLUX.1-schnell
HF_API_BASE_URL=https://router.huggingface.co/hf-inference/models

ANTHROPIC_API_KEY=your_anthropic_key
PROMPT_REFINER_MODEL=claude-3-haiku-20240307
```

Notes:

- `HUGGINGFACE_API_KEY` is required for image generation.
- `ANTHROPIC_API_KEY` is optional and only used for prompt refinement.
- `PROMPT_REFINER_MODEL` is optional. If omitted, the app falls back to `MODEL_NAME` or a default Claude model.
- `MODEL_NAME` is no longer needed for scene segmentation because NLTK handles that step.

### 4. Run the app

```powershell
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## How It Works

1. The user pastes a story into the input form.
2. FastAPI receives the form submission at `/generate`.
3. NLTK tokenizes the text into sentences.
4. The app selects 3 to 5 key scenes.
5. A prompt is built for each scene with a chosen visual style.
6. If Anthropic is configured, each prompt is optionally refined into a more vivid image prompt.
7. Hugging Face generates an image for each prompt.
8. Jinja2 renders the storyboard page with the images and captions.

## Prompt Engineering Approach

The project uses a layered prompt strategy:

- NLTK provides the narrative segments.
- A style preset adds a consistent artistic direction to every scene.
- A shared prompt suffix keeps the storyboard visually coherent.
- Optional Anthropic refinement expands short scene prompts into stronger image prompts with composition, lighting, and atmosphere details.

This approach keeps the system simple and deterministic at the segmentation stage while still allowing richer prompt generation when a secondary LLM is available.

## Design Choices

- **NLTK for segmentation**: lightweight, easy to run, and satisfies the sentence-tokenization requirement.
- **FastAPI + Jinja2**: simple server-side rendering with minimal frontend complexity.
- **Hugging Face for images**: keeps the image generation step external and easy to swap.
- **Style dropdown**: allows the user to guide the storyboard look without changing code.
- **Fallback SVG panels**: if image generation fails, the storyboard still renders cleanly.

## API Key Management

Keep secrets out of version control.

- Store keys in `.env`.
- Do not commit `.env`.
- Use `.gitignore` to exclude secret files.

Required for full functionality:

- `HUGGINGFACE_API_KEY`

Optional:

- `ANTHROPIC_API_KEY`
- `PROMPT_REFINER_MODEL`
- `HF_IMAGE_MODEL`
- `HF_API_BASE_URL`

## Troubleshooting

- If no images appear, check that your Hugging Face credits are active.
- If prompt refinement does not run, verify `ANTHROPIC_API_KEY` is set.
- If sentence splitting looks off, make sure NLTK packages are installed and the app has internet access for the initial tokenizer download if needed.

## Submission Note

This repository is designed to be submitted as a single GitHub repository containing runnable source code. After pushing the project to GitHub, replace this section with the repository URL:

```text
https://github.com/<your-username>/pitch_visualizer
```

## License

Add a license if required by your submission rules.
