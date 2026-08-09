"""
FastAPI Backend - Zero-Shot Voice Cloning API.

This module exposes the cloning pipeline over HTTP. The ``/api/clone``
endpoint accepts a reference audio file and a target text, then runs the
sequential DSP -> ASR -> TTS pipeline and returns the synthesized WAV file.

Key implementation details
--------------------------
* The TTS and ASR models are instantiated once at startup and reused across
  requests (singleton pattern) to avoid re-downloading/re-loading heavy
  models on every call.
* Uploaded files are written to a temp working directory before processing.
* CORS is enabled so the Streamlit UI (running on a different port) can call
  the API.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Local modules
from src.audio.cleaner import preprocess_audio
from src.audio.transcriber import WhisperTranscriber
from src.models.tts_engine import VoiceClonerEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("voice-cloner.api")

# ----------------------------- App setup --------------------------------- #

app = FastAPI(
    title="Zero-Shot Voice Cloning Engine",
    version="0.1.0",
    description=(
        "Pipeline: DSP cleaning -> Whisper ASR -> zero-shot TTS "
        "(F5-TTS / XTTS-v2). Upload reference audio + target text to clone."
    ),
)

# Allow the Streamlit UI (and any origin) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Strings that are safe to expose to the client (no credentials).
TEMP_DIR = Path(tempfile.gettempdir()) / "voice-cloner"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- Model singletons ------------------------------ #

# Instantiate once. TTS/ASR models are lazily loaded inside the classes, so
# app startup stays fast; first request will trigger actual model loading.
transcriber = WhisperTranscriber(model_name="base")  # whisper 'base'
engine = VoiceClonerEngine()  # auto-detects f5-tts / xtts / mock


# ------------------------------ Helpers ---------------------------------- #


def _save_upload(upload: UploadFile) -> Path:
    """Persist an uploaded file to the temp working directory."""
    ext = Path(upload.filename or "ref.wav").suffix or ".wav"
    dest = TEMP_DIR / f"{uuid.uuid4().hex}_ref{ext}"
    with dest.open("wb") as fh:
        fh.write(upload.file.read())
    logger.info("Saved upload -> %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def _run_clone_pipeline(ref_path: Path, target_text: str) -> Path:
    """
    Execute clean -> transcribe -> synthesize and return output WAV path.
    """
    # 1) DSP preprocessing
    cleaned_path = TEMP_DIR / f"{ref_path.stem}_cleaned.wav"
    preprocess_audio(ref_path, cleaned_path, target_sr=24000)

    # 2) Transcribe reference -> ref_text
    ref_text = transcriber.transcribe(cleaned_path)
    logger.info("Reference transcript: '%s'", ref_text[:120])
    if not ref_text:
        raise RuntimeError("Transcription returned empty text for reference audio.")

    # 3) Zero-shot synthesis
    output_path = TEMP_DIR / f"{uuid.uuid4().hex}_cloned.wav"
    engine.synthesize(
        ref_audio_path=cleaned_path,
        ref_text=ref_text,
        target_text=target_text,
        output_path=output_path,
    )
    return output_path


# ------------------------------ Routes ----------------------------------- #


@app.get("/health")
def health() -> dict:
    """Simple liveness probe."""
    return {
        "status": "ok",
        "service": "zero-shot-voice-cloner",
        "backend": engine.backend,
        "device": engine.device,
    }


@app.post("/api/clone")
async def clone_voice(
    ref_audio: UploadFile = File(...),
    target_text: str = Form(...),
):
    """
    Clone a voice.

    Accepts a multipart form with:
      * ``ref_audio``  : reference audio file upload (UploadFile).
      * ``target_text``: the text to synthesize in the cloned voice.

    Returns the cloned audio as a WAV FileResponse.
    """
    ref_text: Optional[str] = None
    try:
        ref_path = _save_upload(ref_audio)
        audio_path = _run_clone_pipeline(ref_path, target_text)
        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename="cloned_voice.wav",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - return a clean 500 for the UI
        logger.exception("Cloning failed")
        raise HTTPException(
            status_code=500, detail=f"Cloning failed: {exc}"
        ) from exc
    finally:
        # Best-effort cleanup of large temp files.
        for p in TEMP_DIR.glob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except OSError:  # pragma: no cover
                pass


@app.on_event("startup")
async def startup_log() -> None:
    """Log the effective configuration at startup."""
    logger.info(
        "Voice Cloner API started (backend='%s', device='%s', model='%s')",
        engine.backend,
        engine.device,
        engine.model_name,
    )


__all__ = ["app"]

