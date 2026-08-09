"""
API sub-package.

Exposes the Zero-Shot Voice Cloning pipeline over HTTP using FastAPI.
"""

from .main import app

__all__ = ["app"]
