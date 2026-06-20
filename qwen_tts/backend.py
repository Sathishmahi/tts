"""
Qwen3-TTS — FastAPI Backend (0.6B-only build)
Trimmed for low-VRAM laptops (e.g. RTX 3050 6GB / 16GB RAM).

Modes:
  • Custom Voice  (9 preset speakers) — Qwen3-TTS-12Hz-0.6B-CustomVoice
  • Voice Design  (text -> brand-new voice) — Qwen3-TTS-12Hz-1.7B-VoiceDesign
        NOTE: VoiceDesign has NO 0.6B checkpoint upstream. This is the only
        size available for this mode, so it will use more VRAM than the
        other two. If you hit OOM, skip this tab or free other GPU memory.
  • Voice Clone   (clone from 3+s reference audio) — Qwen3-TTS-12Hz-0.6B-Base

Run with:  python backend.py
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import io
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Model registry ─────────────────────────────────────────────────────────────
# Pared down to fit a 6GB VRAM / 16GB RAM laptop.
# Custom Voice & Voice Clone use the small 0.6B checkpoints.
# Voice Design has no 0.6B checkpoint upstream — 1.7B is the only option for it.
MODEL_IDS = {
    ("custom", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ("design", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ("clone", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

_model_cache: dict[str, "Qwen3TTSModel"] = {}  # noqa: F821


def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def get_model(mode: str, size: str):
    """Lazily load (and cache) the checkpoint for a given mode/size."""
    key = (mode, size)
    if key not in MODEL_IDS:
        raise HTTPException(status_code=422, detail=f"No checkpoint for mode='{mode}' size='{size}'.")

    model_id = MODEL_IDS[key]
    if model_id in _model_cache:
        return _model_cache[model_id]

    from qwen_tts import Qwen3TTSModel

    device, dtype = get_device_and_dtype()
    logger.info(f"Loading {model_id} on {device} ({dtype})...")

    try:
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            attn_implementation="flash_attention_2" if device.startswith("cuda") else "sdpa",
        )
    except Exception:
        # Fall back gracefully if flash-attn isn't installed
        model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype)

    _model_cache[model_id] = model
    logger.info(f"✅ {model_id} loaded.")
    return model


# ─── App ────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm only the smallest/most common checkpoint on startup.
    try:
        get_model("custom", "0.6B")
    except Exception as exc:
        logger.warning(f"Pre-warm skipped: {exc}")
    yield


app = FastAPI(
    title="Qwen3-TTS API (0.6B build)",
    description=(
        "Custom Voice & Voice Clone on the 0.6B checkpoints, "
        "Voice Design on 1.7B (no 0.6B checkpoint exists for it) — QwenLM/Qwen3-TTS"
    ),
    version="1.0.0-0.6b",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_RATE = 24000

# Static reference data (mirrors the official README)
SPEAKERS = {
    "Vivian":   {"gender": "Female", "native_language": "Chinese",                  "description": "Bright, slightly edgy young female voice."},
    "Serena":   {"gender": "Female", "native_language": "Chinese",                  "description": "Warm, gentle young female voice."},
    "Uncle_fu": {"gender": "Male",   "native_language": "Chinese",                  "description": "Seasoned male voice with a low, mellow timbre."},
    "Dylan":    {"gender": "Male",   "native_language": "Chinese (Beijing dialect)","description": "Youthful Beijing male voice with a clear, natural timbre."},
    "Eric":     {"gender": "Male",   "native_language": "Chinese (Sichuan dialect)","description": "Lively Chengdu male voice with a slightly husky brightness."},
    "Ryan":     {"gender": "Male",   "native_language": "English",                  "description": "Dynamic male voice with strong rhythmic drive."},
    "Aiden":    {"gender": "Male",   "native_language": "English",                  "description": "Sunny American male voice with a clear midrange."},
    "Ono_anna": {"gender": "Female", "native_language": "Japanese",                 "description": "Playful Japanese female voice with a light, nimble timbre."},
    "Sohee":    {"gender": "Female", "native_language": "Korean",                   "description": "Warm Korean female voice with rich emotion."},
}

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]


def audio_to_wav_bytes(wav, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def build_gen_kwargs(max_new_tokens, top_p, top_k, temperature, repetition_penalty) -> dict:
    kwargs = {}
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    if temperature is not None:
        kwargs["temperature"] = temperature
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty
    return kwargs


# ─── Meta endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "loaded_models": list(_model_cache.keys())}


@app.get("/speakers")
async def speakers():
    return SPEAKERS


@app.get("/languages")
async def languages():
    return LANGUAGES


# ─── 1) Custom Voice — 0.6B only ────────────────────────────────────────────────
@app.post("/generate/custom-voice", response_class=Response)
async def generate_custom_voice(
    text: str = Form(...),
    speaker: str = Form("Ryan", description="One of the 9 preset speakers."),
    language: str = Form("Auto"),
    max_new_tokens: Optional[int] = Form(2048),
    top_p: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
):
    """
    0.6B CustomVoice checkpoint only.
    Note: the 0.6B checkpoint does not support the natural-language
    `instruct` style control — that feature is 1.7B-only.
    """
    if speaker not in SPEAKERS:
        raise HTTPException(status_code=422, detail=f"Unknown speaker '{speaker}'. Choose from {list(SPEAKERS)}")

    model = get_model("custom", "0.6B")
    kwargs = build_gen_kwargs(max_new_tokens, top_p, top_k, temperature, repetition_penalty)

    try:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct="",
            **kwargs,
        )
        return Response(
            content=audio_to_wav_bytes(wavs[0], sr),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=custom_voice.wav"},
        )
    except Exception as exc:
        logger.exception("custom-voice generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─── 2) Voice Design — 1.7B (no 0.6B checkpoint exists) ────────────────────────
@app.post("/generate/voice-design", response_class=Response)
async def generate_voice_design(
    text: str = Form(...),
    instruct: str = Form(..., description="Free-text description of the voice you want."),
    language: str = Form("Auto"),
    max_new_tokens: Optional[int] = Form(2048),
    top_p: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
):
    """
    Always uses the 1.7B VoiceDesign checkpoint — Qwen does not publish a
    0.6B VoiceDesign model. This is the heaviest of the three modes on VRAM.
    """
    if not instruct.strip():
        raise HTTPException(status_code=422, detail="instruct is required for Voice Design.")

    model = get_model("design", "1.7B")
    kwargs = build_gen_kwargs(max_new_tokens, top_p, top_k, temperature, repetition_penalty)

    try:
        wavs, sr = model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            **kwargs,
        )
        return Response(
            content=audio_to_wav_bytes(wavs[0], sr),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=voice_design.wav"},
        )
    except Exception as exc:
        logger.exception("voice-design generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─── 3) Voice Clone — 0.6B only ─────────────────────────────────────────────────
@app.post("/generate/voice-clone", response_class=Response)
async def generate_voice_clone(
    text: str = Form(...),
    ref_text: Optional[str] = Form(None, description="Transcript of ref_audio. Required unless x_vector_only_mode=True."),
    ref_audio_url: Optional[str] = Form(None, description="URL to reference audio (alternative to file upload)."),
    ref_audio: Optional[UploadFile] = File(None, description="Reference audio file (alternative to URL)."),
    language: str = Form("Auto"),
    x_vector_only_mode: bool = Form(False, description="Skip transcript, use only the speaker embedding (faster, lower quality)."),
    max_new_tokens: Optional[int] = Form(2048),
    top_p: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
):
    """0.6B Base checkpoint only."""
    if not ref_audio and not ref_audio_url:
        raise HTTPException(status_code=422, detail="Provide either ref_audio (file) or ref_audio_url.")
    if not x_vector_only_mode and not (ref_text and ref_text.strip()):
        raise HTTPException(
            status_code=422,
            detail="ref_text is required in ICL mode. Either provide a transcript or enable x_vector_only_mode.",
        )

    tmp_path = None
    try:
        if ref_audio is not None:
            audio_bytes = await ref_audio.read()
            suffix = os.path.splitext(ref_audio.filename or ".wav")[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            ref_audio_input = tmp_path
        else:
            ref_audio_input = ref_audio_url  # qwen-tts accepts URLs directly

        model = get_model("clone", "0.6B")
        kwargs = build_gen_kwargs(max_new_tokens, top_p, top_k, temperature, repetition_penalty)

        wavs, sr = model.generate_voice_clone(
            text=text,
            ref_audio=ref_audio_input,
            ref_text=(ref_text.strip() if ref_text and not x_vector_only_mode else None),
            x_vector_only_mode=x_vector_only_mode,
            language=language,
            **kwargs,
        )
        return Response(
            content=audio_to_wav_bytes(wavs[0], sr),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=voice_clone.wav"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("voice-clone generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─── Unload (free VRAM) ─────────────────────────────────────────────────────────
@app.post("/unload")
async def unload_models():
    _model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "all models unloaded"}


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False, log_level="info")