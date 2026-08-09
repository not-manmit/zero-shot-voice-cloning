"""
Audio processing sub-package.

Contains the DSP (cleaner) and ASR (transcriber) modules used to prepare
reference audio before zero-shot voice cloning.
"""

from .cleaner import preprocess_audio
from .transcriber import WhisperTranscriber

__all__ = ["preprocess_audio", "WhisperTranscriber"]
