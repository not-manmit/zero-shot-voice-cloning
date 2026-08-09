"""
Zero-Shot Voice Cloning System - Source Package.

This package contains the modular layers of the voice cloning pipeline:

- ``audio``     : DSP (cleaner) and ASR (transcriber) modules.
- ``models``    : Zero-shot TTS inference wrapper (F5-TTS / XTTS-v2).
- ``api``       : FastAPI backend exposing the cloning workflow over HTTP.
"""

__version__ = "0.1.0"
