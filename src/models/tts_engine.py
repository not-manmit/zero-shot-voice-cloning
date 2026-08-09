"""
Inference Layer - Zero-Shot TTS Engine.

This module implements :class:`VoiceClonerEngine`, a production-oriented
wrapper around open-source zero-shot text-to-speech models (F5-TTS and
XTTS-v2). It abstracts model loading, device placement, and synthesis into a
simple ``synthesize`` API.

Design decisions
----------------
* **Lazy loading**: The heavy model is loaded into GPU/CPU memory only on
  first use (or when :meth:`load` is called explicitly), keeping process
  startup fast and avoiding OOM on import.
* **Backend abstraction**: The engine supports multiple backends. The
  ``f5-tts`` backend is preferred when the package is installed; otherwise it
  falls back to ``xtts`` or a clearly-marked ``mock`` backend that emits a
  WAV file so the full pipeline can be tested end-to-end without a GPU.
* **Device handling**: Automatically selects CUDA when available, with CPU
  and MPS fallbacks for portability.
"""

from __future__ import annotations

import logging
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SynthesizeResult:
    """Result of a ``synthesize`` call."""

    output_path: Path
    backend: str
    duration_s: float
    sample_rate: int


def resolve_device(prefer: Optional[str] = None) -> str:
    """
    Resolve the compute device to use for TTS inference.

    Args:
        prefer: Optional preferred device string (``"cuda"``, ``"cpu"``,
            ``"mps"``). If ``None``, auto-detect.

    Returns:
        A valid torch device string.
    """
    if prefer:
        return prefer

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _detect_backend() -> str:
    """Return the best available TTS backend key."""
    try:
        import f5_tts  # noqa: F401

        return "f5-tts"
    except ImportError:
        pass
    try:
        import TTS  # noqa: F401  (Coqui XTTS)

        return "xtts"
    except ImportError:
        pass
    logger.warning(
        "No zero-shot TTS backend (f5-tts/XTTS) installed. Using 'mock' backend."
    )
    return "mock"


class VoiceClonerEngine:
    """
    Wrapper for zero-shot voice cloning TTS models.

    Args:
        backend: TTS backend key (``"f5-tts"``, ``"xtts"``, ``"mock"``).
            If ``None``, the best available backend is auto-detected.
        device: Torch device string. If ``None``, auto-resolved.
        model_name: Backend-specific model identifier (e.g. an F5-TTS model
            name or XTTS model path).
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.backend = backend or _detect_backend()
        self.device = resolve_device(device)
        self.model_name = model_name or "default"
        self._engine = None  # lazily-loaded backend handle
        self._sample_rate = self._infer_sample_rate()

        logger.info(
            "VoiceClonerEngine initialized (backend='%s', device='%s', model='%s')",
            self.backend,
            self.device,
            self.model_name,
        )

    @staticmethod
    def _infer_sample_rate() -> int:
        """Sample rate expected by the TTS backends (Hz)."""
        return 24000

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Load the underlying TTS model into memory (idempotent)."""
        if self._engine is not None:
            return

        logger.info("Loading '%s' backend into memory...", self.backend)
        if self.backend == "f5-tts":
            self._engine = self._load_f5tts()
        elif self.backend == "xtts":
            self._engine = self._load_xtts()
        else:
            self._engine = self._load_mock()
        logger.info("Backend '%s' loaded successfully.", self.backend)

    def _load_f5tts(self):
        """Load F5-TTS model (best effort; adapt to the installed API)."""
        try:
            from f5_tts.api import F5TTS  # type: ignore

            # GPU memory offload is handled by the framework; we pass the
            # resolved device where supported.
            return F5TTS(model="F5TTS_v1_Base", device=self.device)
        except Exception as exc:  # pragma: no cover - API drift tolerance
            logger.error("Failed to load F5-TTS backend: %s", exc)
            raise

    def _load_xtts(self):
        """Load Coqui XTTS-v2 model."""
        try:
            from TTS.api import TTS  # type: ignore

            return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
                self.device
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load XTTS backend: %s", exc)
            raise

    def _load_mock(self):
        """Return a lightweight mock backend for end-to-end testing."""
        return {"mock": True}

    # ------------------------------------------------------------------ #
    # Core synthesis API
    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        ref_audio_path: str | Path,
        ref_text: str,
        target_text: str,
        output_path: str | Path,
    ) -> Path:
        """
        Synthesize speech in the reference voice speaking ``target_text``.

        Args:
            ref_audio_path: Path to the cleaned reference audio clip.
            ref_text: Transcript of the reference audio (from Whisper).
            target_text: The text to be spoken in the cloned voice.
            output_path: Destination path for the generated WAV file.

        Returns:
            Path to the generated WAV file.

        Raises:
            RuntimeError: If the backend fails to generate audio.
        """
        self.load()  # ensure model is resident

        ref_audio_path = Path(ref_audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        logger.info(
            "Synthesizing with backend='%s' (ref=%s, text='%s')",
            self.backend,
            ref_audio_path.name,
            target_text[:60],
        )

        if self.backend == "f5-tts":
            audio, sr = self._infer_f5tts(ref_audio_path, ref_text, target_text)
        elif self.backend == "xtts":
            audio, sr = self._infer_xtts(ref_audio_path, ref_text, target_text)
        else:
            audio, sr = self._infer_mock(output_path)

        self._write_wav(audio, sr, output_path)
        logger.info("Wrote cloned audio to %s", output_path)
        return output_path

    # ------------------------------------------------------------------ #
    # Backend-specific inference
    # ------------------------------------------------------------------ #
    def _infer_f5tts(
        self, ref_audio_path: Path, ref_text: str, target_text: str
    ) -> tuple[np.ndarray, int]:
        """Run F5-TTS inference, returning ``(audio, sr)``."""
        engine = self._engine
        # F5-TTS API: engine.infer(ref_file, ref_text, gen_text)
        result = engine.infer(
            ref_file=str(ref_audio_path),
            ref_text=ref_text,
            gen_text=target_text,
        )
        # The API returns a tuple (waveform, sr, ...) depending on version.
        if isinstance(result, tuple):
            audio = result[0]
            sr = int(result[1]) if len(result) > 1 and result[1] else self._sample_rate
        else:
            audio = result
            sr = self._sample_rate
        return self._as_float32_mono(audio), sr

    def _infer_xtts(
        self, ref_audio_path: Path, ref_text: str, target_text: str
    ) -> tuple[np.ndarray, int]:
        """Run XTTS-v2 inference, returning ``(audio, sr)``."""
        engine = self._engine
        sr = 24000
        wav = engine.tts(text=target_text, speaker_wav=str(ref_audio_path), language="en")
        return self._as_float32_mono(wav), sr

    def _infer_mock(self, output_path: Path) -> tuple[np.ndarray, int]:
        """Generate a placeholder sine-wave WAV so the pipeline is testable."""
        sr = self._sample_rate
        duration = 2.0
        t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
        audio = 0.3 * np.sin(2.0 * np.pi * 180.0 * t)
        logger.warning(
            "Mock backend produced placeholder audio -> %s", output_path
        )
        return audio.astype(np.float32), sr

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_float32_mono(audio) -> np.ndarray:
        """Normalize a backend audio tensor/np array to float32 mono."""
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def _write_wav(audio: np.ndarray, sr: int, output_path: Path) -> None:
        """Write a float [-1,1] array as a 16-bit PCM WAV file."""
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(int(sr))
            wf.writeframes(pcm.tobytes())

    def unload(self) -> None:
        """Release GPU memory held by the backend (best effort)."""
        if self.backend == "f5-tts" and self._engine is not None:
            try:
                del self._engine
            except Exception:
                pass
        self._engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Engine resources released.")


__all__ = ["VoiceClonerEngine", "SynthesizeResult", "resolve_device"]
