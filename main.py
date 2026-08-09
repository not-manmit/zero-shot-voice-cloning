"""
Local entry point for the Zero-Shot Voice Cloning FastAPI server.

Usage:
    python main.py                     # start on 0.0.0.0:8000
    python main.py --port 9000         # custom port
    python main.py --reload            # enable auto-reload (dev)

The Streamlit UI (``ui/app.py``) can then point at
``http://localhost:8000/api/clone``.
"""

from __future__ import annotations

import argparse
import logging
import os
import uvicorn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("voice-cloner.main")

DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("PORT", "8000"))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the server."""
    parser = argparse.ArgumentParser(description="Zero-Shot Voice Cloning API")
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="Bind host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Bind port (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable uvicorn auto-reload (dev)"
    )
    return parser.parse_args()


def main() -> None:
    """Launch the FastAPI app with Uvicorn."""
    args = parse_args()
    logger.info("Starting Voice Cloner API on %s:%d", args.host, args.port)
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

