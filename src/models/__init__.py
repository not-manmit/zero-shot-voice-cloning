"""
Model inference sub-package.

Contains the zero-shot TTS engine wrapper used to synthesize speech in a
target voice conditioned on a reference audio clip and its transcript.
"""

from .tts_engine import VoiceClonerEngine

__all__ = ["VoiceClonerEngine"]
