"""
DSP Layer - Audio Cleaner.

This module provides signal-processing utilities that prepare raw reference
audio for zero-shot voice cloning. The primary entry point is
:func:`preprocess_audio`, which runs a deterministic chain of DSP operations:

    1. Load audio with Librosa (auto-resamples to ``target_sr``).
    2. Down-mix stereo/multi-channel audio to mono.
    3. Apply spectral-gating noise suppression via ``noisereduce``.
    4. Trim leading/trailing silence (with a short keep-padding).
    5. Peak-normalize the signal to a ``[-1.0, 1.0]`` range.

The output is written as a 16-bit PCM WAV file (or the format inferred from
``output_path``) that downstream modules (ASR + TTS) can consume reliably.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def load_audio(
    input_path: str | Path,
    target_sr: int = 24000,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file, resampling to ``target_sr`` and down-mixing to mono.

    Args:
        input_path: Path to the input audio file (any format Librosa supports).
        target_sr: Sample rate to resample to (Hz).
        mono: If True, down-mix all channels into a single channel.

    Returns:
        A tuple ``(audio, sr)`` where ``audio`` is a ``float32`` numpy array
        in ``[-1.0, 1.0]`` and ``sr`` is the effective sample rate.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    # Load with resampling. Using mono=True here gives a fast first-pass mono,
    # but we keep the explicit down-mix logic below for full channel control.
    audio, sr = librosa.load(str(input_path), sr=target_sr, mono=mono)

    if audio.ndim > 1:
        # Safety net: explicitly down-mix any remaining multi-channel audio.
        logger.info("Down-mixing %d channels to mono", audio.shape[0])
        audio = librosa.to_mono(audio)

    # Clip to the valid float32 range to avoid NaNs/overflow in later stages.
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    logger.info("Loaded audio: sr=%d, duration=%.2fs", sr, audio.shape[0] / sr)
    return audio, sr


def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply spectral-gating noise suppression to ``audio``.

    The first ``noise_floor_duration`` seconds of the clip are used as the
    noise profile sample. If the clip is shorter than that window, the whole
    clip is used as a conservative fallback.

    Args:
        audio: Mono ``float32`` audio signal in ``[-1.0, 1.0]``.
        sr: Sample rate of ``audio``.

    Returns:
        Noise-reduced audio signal (same shape/dtype as input).
    """
    noise_floor_duration = 0.5  # seconds of leading audio used as noise profile
    noise_samples = min(int(noise_floor_duration * sr), max(audio.shape[0] - 1, 1))
    noise_profile = audio[:noise_samples]

    logger.info("Applying spectral-gating noise reduction...")
    reduced = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_profile,
        prop_decrease=0.85,
        stationary=True,
    )
    return np.clip(reduced, -1.0, 1.0).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: float = 30.0,
    keep_pad_s: float = 0.25,
) -> np.ndarray:
    """
    Trim leading and trailing silence using Librosa's energy-based detector.

    Args:
        audio: Mono ``float32`` audio signal.
        sr: Sample rate of ``audio``.
        top_db: Threshold (dB) below the peak considered as silence.
        keep_pad_s: Seconds of padding kept around the trimmed boundaries
            to avoid clipping the first/last phonemes.

    Returns:
        The trimmed audio signal.
    """
    if audio.shape[0] == 0:
        return audio

    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)

    # Re-apply a short fade-in/fade-out padding around boundaries so the
    # trimmed clip does not start/end abruptly (helps the TTS reference).
    pad_len = int(keep_pad_s * sr)
    if pad_len > 0 and trimmed.shape[0] > 2 * pad_len:
        trimmed = np.pad(trimmed, pad_len, mode="reflect")

    logger.info(
        "Trimmed silence: %.2fs -> %.2fs",
        audio.shape[0] / sr,
        trimmed.shape[0] / sr,
    )
    return np.clip(trimmed, -1.0, 1.0).astype(np.float32)


def peak_normalize(audio: np.ndarray, peak: float = 1.0) -> np.ndarray:
    """
    Peak-normalize ``audio`` so its maximum absolute amplitude equals ``peak``.

    Args:
        audio: Mono ``float32`` audio signal.
        peak: Target peak amplitude (default ``1.0`` for full-scale).

    Returns:
        Peak-normalized audio signal.
    """
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs < 1e-12:
        logger.warning("Silent audio; skipping peak normalization")
        return audio

    normalized = (audio / max_abs) * peak
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def save_audio(
    audio: np.ndarray,
    output_path: str | Path,
    sr: int,
    subtype: str = "PCM_16",
) -> Path:
    """
    Persist ``audio`` to disk using SoundFile (16-bit PCM WAV by default).

    Args:
        audio: Mono ``float32`` audio signal in ``[-1.0, 1.0]``.
        output_path: Destination file path.
        sr: Sample rate to embed in the file header.
        subtype: SoundFile subtype (e.g. ``"PCM_16"``, ``"FLOAT"``).

    Returns:
        The resolved output :class:`~pathlib.Path`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write as float in [-1,1]; SoundFile converts to the requested subtype.
    sf.write(str(output_path), audio, sr, subtype=subtype)

    logger.info("Saved audio to %s (sr=%d)", output_path, sr)
    return output_path


def preprocess_audio(
    input_path: str | Path,
    output_path: str | Path,
    target_sr: int = 24000,
) -> Path:
    """
    Run the full DSP preprocessing chain on an input audio file.

    Pipeline order: load/resample -> mono -> noise reduction -> silence trim
    -> peak normalization -> save.

    Args:
        input_path: Source audio file path.
        output_path: Destination path for the cleaned audio (e.g. ``.wav``).
        target_sr: Target sample rate used by the TTS reference (Hz).

    Returns:
        The output :class:`~pathlib.Path` where the cleaned audio was saved.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        RuntimeError: If the cleaned audio is silent/empty.
    """
    logger.info("Preprocessing %s (target_sr=%d)", input_path, target_sr)

    audio, sr = load_audio(input_path, target_sr=target_sr, mono=True)
    audio = reduce_noise(audio, sr)
    audio = trim_silence(audio, sr)
    audio = peak_normalize(audio, peak=1.0)

    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-6:
        raise RuntimeError(
            f"Cleaned audio is silent or empty: {input_path}"
        )

    return save_audio(audio, output_path, sr)


# Module-level convenience alias (optional, kept for flexible imports).
clean_reference = preprocess_audio

__all__ = [
    "preprocess_audio",
    "clean_reference",
    "load_audio",
    "reduce_noise",
    "trim_silence",
    "peak_normalize",
    "save_audio",
]
