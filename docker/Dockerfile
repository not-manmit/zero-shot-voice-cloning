# syntax=docker/dockerfile:1

# =====================================================================
# Zero-Shot Voice Cloning - GPU-capable runtime image.
#
# Build:
#   docker build -f docker/Dockerfile -t voice-cloner .
# Run:
#   docker run --gpus all -p 8000:8000 voice-cloner
# =====================================================================

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Avoid interactive prompts during apt installs.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: FFmpeg (audio codecs) + build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Application source.
COPY src ./src
COPY main.py ./main.py
COPY ui ./ui
COPY notebooks ./notebooks

# Expose the FastAPI port.
EXPOSE 8000

# Default command: launch the API server.
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
