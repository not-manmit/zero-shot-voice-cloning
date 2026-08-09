"""
ASR Layer - Reference Audio Transcriber.

This module wraps OpenAI Whisper to convert a cleaned reference audio clip
into its transcript. The transcript is used as the ``ref_text`` condition for
zero-shot voice cloning (the TTS engine uses the reference *voice* from the
audio and the reference *content* from the transcript).

The wrapper is designed to be used as a long-lived singleton so the Whisper
model is loaded into memory only once per process.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import whisper

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Structured result of a Whisper transcription."""

    text: str
    language: Optional[str] = None
    duration_s: Optional[float] = None
    segments: list = field(default_factory=list)


class WhisperTranscriber:
    """
    Thin wrapper around OpenAI Whisper for transcribing reference audio.

    Args:
        model_name: Whisper model size to load (``"base"``, ``"small"``,
            ``"medium"``, ``"large-v3"``...). Defaults to ``"base"`` as a
            good accuracy/speed trade-off.
        device: Torch device string (``"cuda"``, ``"cpu"``, ``"mps"``...).
            If ``None``, CUDA is used when available.
    """

    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or self._auto_device()
        self._model = None  # lazy-loaded model handle
        logger.info(
            "WhisperTranscriber initialized (model='%s', device='%s')",
            self.model_name,
            self.device,
        )

    @staticmethod
    def _auto_device() -> str:
        """Return the best available compute device."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @property
    def model(self):
        """Lazily load and cache the Whisper model on first use."""
        if self._model is None:
            logger.info("Loading Whisper model '%s' on '%s'...", self.model_name, self.device)
            self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def transcribe_file(self, audio_path: str | Path, **whisper_kwargs) -> TranscriptionResult:
        """
        Transcribe an audio file and return a :class:`TranscriptionResult`.

        Args:
            audio_path: Path to the (ideally pre-cleaned) audio file.
            **whisper_kwargs: Extra options forwarded to ``whisper.transcribe``
                (e.g. ``language``, ``fp16``, ``verbose``).

        Returns:
            A :class:`TranscriptionResult` containing the transcript text.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing %s ...", audio_path)
        result = self.model.transcribe(str(audio_path), **whisper_kwargs)

        text = (result.get("text") or "").strip()
        language = result.get("language")
        duration = result.get("duration")
        segments = result.get("segments", [])

        logger.info(
            "Transcription complete (language=%s, duration=%.2fs): '%s'",
            language,
            duration or 0.0,
            text[:80],
        )
        return TranscriptionResult(
            text=text,
            language=language,
            duration_s=duration,
            segments=list(segments),
        )

    def transcribe(self, audio_path: str | Path, **whisper_kwargs) -> str:
        """
        Convenience wrapper returning only the plain transcript text.

        Args:
            audio_path: Path to the audio file to transcribe.
            **whisper_kwargs: Extra options forwarded to Whisper.

        Returns:
            The stripped transcript string.
        """
        return self.transcribe_file(audio_path, **whisper_kwargs).text


def transcribe_reference(
    audio_path: str | Path,
    model_name: str = "base",
    device: Optional[str] = None,
) -> str:
    """
    Functional API: transcribe a reference audio file and return its text.

    This is a convenience function that creates a temporary transcriber for
    one-shot usage. For repeated calls, prefer instantiating
    :class:`WhisperTranscriber` once and reusing it.

    Args:
        audio_path: Path to the cleaned reference audio file.
        model_name: Whisper model size.
        device: Optional device override.

    Returns:
        The transcript text of the reference audio.
    """
    transcriber = WhisperTranscriber(model_name=model_name, device=device)
    return transcriber.transcribe(audio_path)


__all__ = ["WhisperTranscriber", "transcribe_reference", "TranscriptionResult"]
