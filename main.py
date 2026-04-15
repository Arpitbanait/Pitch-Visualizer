import logging
import os
import base64
import asyncio
from dataclasses import dataclass

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx
import nltk
from nltk.tokenize import sent_tokenize

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("pitch-visualizer")


class HFCreditsExhaustedError(RuntimeError):
    pass

app = FastAPI(title="Pitch Visualizer", version="1.0.0")
templates = Jinja2Templates(directory="templates")

MAX_INPUT_CHARS = 12000
DEFAULT_STYLE_SUFFIX = "cinematic storyboard, digital art, dramatic lighting, high detail"
MAX_IMAGE_PROMPT_CHARS = 320
MIN_SCENES = 3
MAX_SCENES = 5
DEFAULT_STYLE_KEY = "digital_art"
STYLE_PRESETS = {
    "digital_art": {
        "label": "Digital Art",
        "suffix": "digital art, cinematic composition, dramatic lighting, rich color grading",
    },
    "photorealistic": {
        "label": "Photorealistic",
        "suffix": "photorealistic, natural skin texture, realistic camera depth, high dynamic range",
    },
    "anime": {
        "label": "Anime",
        "suffix": "anime style, expressive characters, clean linework, vibrant cel-shading",
    },
    "watercolor": {
        "label": "Watercolor",
        "suffix": "watercolor painting, soft edges, paper grain texture, pastel tones",
    },
    "comic": {
        "label": "Comic",
        "suffix": "comic book illustration, bold ink outlines, dynamic action framing, halftone accents",
    },
}
DEFAULT_HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
DEFAULT_HF_API_BASE_URL = "https://router.huggingface.co/hf-inference/models"
HF_REQUEST_RETRIES = 2


@dataclass
class Scene:
    caption: str
    prompt: str
    image_url: str
    backup_image_url: str


def get_style_options() -> list[dict[str, str]]:
    return [{"value": key, "label": value["label"]} for key, value in STYLE_PRESETS.items()]


def resolve_style(style_key: str) -> tuple[str, str, str]:
    selected_key = style_key if style_key in STYLE_PRESETS else DEFAULT_STYLE_KEY
    style_data = STYLE_PRESETS[selected_key]
    combined_suffix = f"{style_data['suffix']}, {DEFAULT_STYLE_SUFFIX}"
    return selected_key, style_data["label"], combined_suffix


def normalize_image_prompt(prompt: str) -> str:
    compact = " ".join(prompt.split())
    if len(compact) <= MAX_IMAGE_PROMPT_CHARS:
        return compact
    return compact[:MAX_IMAGE_PROMPT_CHARS].rsplit(" ", 1)[0]


def build_hf_model_name() -> str:
    return (
        os.getenv("HF_IMAGE_MODEL", "").strip()
        or os.getenv("HUGGINGFACE_MODEL", "").strip()
        or DEFAULT_HF_IMAGE_MODEL
    )


def build_svg_data_uri(text: str, caption: str | None = None) -> str:
    safe_text = text.replace("&", "and").replace("<", "").replace(">", "")[:120]
    safe_caption = (caption or "Image unavailable").replace("&", "and").replace("<", "").replace(">", "")[:60]
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='800' height='500' viewBox='0 0 800 500'>
    <defs>
        <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
            <stop offset='0%' stop-color='#e2e8f0'/>
            <stop offset='100%' stop-color='#cbd5e1'/>
        </linearGradient>
    </defs>
    <rect width='800' height='500' fill='url(#g)'/>
    <rect x='48' y='48' width='704' height='404' rx='28' fill='rgba(255,255,255,0.45)' stroke='#94a3b8' stroke-width='2'/>
    <text x='400' y='215' text-anchor='middle' font-family='Arial, sans-serif' font-size='30' fill='#0f172a'>{safe_caption}</text>
    <text x='400' y='275' text-anchor='middle' font-family='Arial, sans-serif' font-size='18' fill='#334155'>{safe_text}</text>
    <text x='400' y='345' text-anchor='middle' font-family='Arial, sans-serif' font-size='16' fill='#475569'>Hugging Face image unavailable</text>
</svg>"""
    encoded_svg = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded_svg}"


def ensure_nltk_tokenizer() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def split_text_into_sentences(input_text: str) -> list[str]:
    normalized = " ".join(input_text.replace("\n", " ").split())
    if not normalized:
        return []

    try:
        ensure_nltk_tokenizer()
        sentences = [s.strip() for s in sent_tokenize(normalized) if s.strip()]
    except Exception as exc:
        logger.warning("NLTK tokenization failed, using basic sentence split: %s", exc)
        sentences = [s.strip() for s in normalized.split(".") if s.strip()]

    if not sentences:
        sentences = [normalized]

    return sentences


def select_scene_sentences(sentences: list[str], max_scenes: int = MAX_SCENES) -> list[str]:
    if len(sentences) <= max_scenes:
        return sentences

    # Evenly sample sentence positions so the storyboard covers beginning, middle, end.
    step = (len(sentences) - 1) / (max_scenes - 1)
    indices: list[int] = []
    for i in range(max_scenes):
        idx = round(i * step)
        if idx not in indices:
            indices.append(idx)

    cursor = 0
    while len(indices) < max_scenes and cursor < len(sentences):
        if cursor not in indices:
            indices.append(cursor)
        cursor += 1

    indices.sort()
    return [sentences[i] for i in indices[:max_scenes]]


def build_scene_prompt(caption: str, style_suffix: str) -> str:
    return normalize_image_prompt(
        "Storyboard scene based on: "
        f"{caption}. Character-focused composition, coherent environment, {style_suffix}."
    )


def build_hf_api_base_url() -> str:
    return (
        os.getenv("HF_API_BASE_URL", "").strip()
        or os.getenv("HUGGINGFACE_API_BASE_URL", "").strip()
        or DEFAULT_HF_API_BASE_URL
    )


def get_hf_api_key() -> str:
    candidate_keys = (
        os.getenv("HUGGINGFACE_API_KEY", "").strip(),
        os.getenv("HF_API_KEY", "").strip(),
        os.getenv("HF_TOKEN", "").strip(),
        os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip(),
    )

    for key in candidate_keys:
        if key:
            return key

    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        with open(dotenv_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                if name.strip() in {"HUGGINGFACE_API_KEY", "HF_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
                    value = value.strip().strip('"').strip("'")
                    if value:
                        return value
    except FileNotFoundError:
        pass

    return ""


def get_prompt_refiner_client() -> Anthropic | None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


def get_prompt_refiner_model() -> str:
    return (
        os.getenv("PROMPT_REFINER_MODEL", "").strip()
        or os.getenv("MODEL_NAME", "").strip()
        or "claude-3-haiku-20240307"
    )


def refine_scene_prompts_with_anthropic(scenes: list[Scene], style_label: str) -> list[Scene]:
    client = get_prompt_refiner_client()
    if client is None:
        return scenes

    model_name = get_prompt_refiner_model()
    for scene in scenes:
        base_prompt = scene.prompt
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=220,
                temperature=0.4,
                system=(
                    "You are a visual prompt engineer for storyboards. "
                    "Return only one concise image prompt, no JSON, no markdown."
                ),
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Style: {style_label}. "
                            f"Caption: {scene.caption}. "
                            f"Current prompt: {base_prompt}. "
                            "Rewrite into one highly visual prompt with composition, lighting, atmosphere, "
                            "and character/action details while staying under 300 characters."
                        ),
                    }
                ],
            )

            text_blocks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
            if text_blocks:
                refined = " ".join(text_blocks).strip().replace("\n", " ")
                if refined:
                    scene.prompt = normalize_image_prompt(refined)
        except Exception as exc:
            logger.warning("Anthropic refinement skipped for scene '%s': %s", scene.caption, exc)

    return scenes


def generate_scenes_with_nltk(input_text: str, style_suffix: str = DEFAULT_STYLE_SUFFIX) -> list[Scene]:
    sentences = split_text_into_sentences(input_text)
    selected = select_scene_sentences(sentences, max_scenes=MAX_SCENES)

    if not selected:
        selected = [input_text.strip()]

    while len(selected) < MIN_SCENES and selected:
        selected.append(selected[-1])

    scenes: list[Scene] = []
    for sentence in selected[:MAX_SCENES]:
        scenes.append(
            Scene(
                caption=sentence,
                prompt=build_scene_prompt(sentence, style_suffix),
                image_url="",
                backup_image_url=build_svg_data_uri("Fallback image", sentence),
            )
        )

    return scenes


async def generate_hf_image_data_uri(prompt: str, caption: str) -> str:
    api_key = get_hf_api_key()
    if not api_key:
        raise RuntimeError("HUGGINGFACE_API_KEY is missing in .env")

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": normalize_image_prompt(prompt),
        "parameters": {
            "negative_prompt": "blurry, low quality, distorted, text, watermark, extra limbs",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        },
        "options": {"wait_for_model": True, "use_cache": False},
    }

    timeout = httpx.Timeout(120.0, connect=30.0)

    request_headers = {
        **headers,
        "Accept": "image/png",
        "Content-Type": "application/json",
    }

    model_name = build_hf_model_name()
    url = f"{build_hf_api_base_url().rstrip('/')}/{model_name}"

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for attempt in range(1, HF_REQUEST_RETRIES + 2):
            try:
                response = await client.post(url, headers=request_headers, json=payload)
            except httpx.ReadTimeout:
                logger.warning(
                    "Hugging Face timeout for '%s' with model %s (attempt %d/%d)",
                    caption,
                    model_name,
                    attempt,
                    HF_REQUEST_RETRIES + 1,
                )
                if attempt <= HF_REQUEST_RETRIES:
                    await asyncio.sleep(1.2 * attempt)
                    continue
                break
            except httpx.HTTPError as exc:
                logger.warning(
                    "Hugging Face transport error for '%s' with model %s: %s",
                    caption,
                    model_name,
                    exc,
                )
                break

            content_type = response.headers.get("content-type", "")
            if response.status_code == 200 and "image" in content_type.lower():
                encoded_image = base64.b64encode(response.content).decode("ascii")
                logger.info("Hugging Face image generation succeeded with model: %s", model_name)
                return f"data:{content_type};base64,{encoded_image}"

            message = response.text[:500]
            if response.status_code == 402:
                logger.warning(
                    "Hugging Face credits exhausted for model %s: %s",
                    model_name,
                    message,
                )
                raise HFCreditsExhaustedError("Hugging Face credits exhausted")

            logger.warning(
                "Hugging Face image generation failed for '%s' with %s (attempt %d/%d): %s",
                caption,
                model_name,
                attempt,
                HF_REQUEST_RETRIES + 1,
                message,
            )
            if attempt <= HF_REQUEST_RETRIES:
                await asyncio.sleep(1.2 * attempt)
                continue
            break

    return build_svg_data_uri("Image generation failed", caption)


async def attach_hf_images(scenes: list[Scene]) -> list[Scene]:
    credits_exhausted = False

    for scene in scenes:
        if credits_exhausted:
            scene.image_url = build_svg_data_uri("Hugging Face credits exhausted", scene.caption)
            scene.backup_image_url = build_svg_data_uri("Fallback image", scene.caption)
            continue

        try:
            scene.image_url = await generate_hf_image_data_uri(scene.prompt, scene.caption)
        except HFCreditsExhaustedError:
            credits_exhausted = True
            scene.image_url = build_svg_data_uri("Hugging Face credits exhausted", scene.caption)
        except Exception as exc:
            logger.warning("Fallback image used for scene '%s': %s", scene.caption, exc)
            scene.image_url = build_svg_data_uri("Image generation failed", scene.caption)
        scene.backup_image_url = build_svg_data_uri("Fallback image", scene.caption)
    return scenes


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "error": None,
            "text": "",
            "style_options": get_style_options(),
            "selected_style": DEFAULT_STYLE_KEY,
        },
    )


@app.post("/generate", response_class=HTMLResponse)
async def generate_storyboard(
    request: Request,
    text: str = Form(...),
    style: str = Form(DEFAULT_STYLE_KEY),
) -> HTMLResponse:
    clean_text = text.strip()
    selected_style, style_label, style_suffix = resolve_style(style)

    if len(clean_text) < 20:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "error": "Please paste a slightly longer story (at least 20 characters).",
                "text": clean_text,
                "style_options": get_style_options(),
                "selected_style": selected_style,
            },
            status_code=400,
        )

    if len(clean_text) > MAX_INPUT_CHARS:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "error": f"Story is too long. Keep it under {MAX_INPUT_CHARS} characters.",
                "text": clean_text,
                "style_options": get_style_options(),
                "selected_style": selected_style,
            },
            status_code=400,
        )

    try:
        scenes = generate_scenes_with_nltk(clean_text, style_suffix=style_suffix)
        scenes = await asyncio.to_thread(refine_scene_prompts_with_anthropic, scenes, style_label)
        logger.info("Generated %d scenes from NLTK tokenization", len(scenes))
        scenes = await attach_hf_images(scenes)
    except Exception as exc:
        logger.exception("Scene generation failed: %s", exc)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "error": "Unable to generate scenes right now. Please try again.",
                "text": clean_text,
                "style_options": get_style_options(),
                "selected_style": selected_style,
            },
            status_code=500,
        )

    return templates.TemplateResponse(
        request,
        "storyboard.html",
        {
            "request": request,
            "scenes": scenes,
            "warning": None,
            "source_text": clean_text,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
