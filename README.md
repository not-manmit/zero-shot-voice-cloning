# 🎙️ Zero-Shot Voice Cloning Engine (DSP + GenAI)

A modular, production-grade boilerplate for a **Zero-Shot Voice Cloning System**
built with **FastAPI**, **PyTorch**, **OpenAI Whisper**, **F5-TTS / XTTS-v2**,
signal processing (**Librosa / noisereduce**), and a **Streamlit** testing UI.

Given a short reference clip (15–30s), the pipeline:

1. **Cleans** the audio (mono, resample, noise gate, silence trim, peak normalize).
2. **Transcribes** it with Whisper to obtain the reference text.
3. **Synthesizes** the target text in the cloned voice using a zero-shot TTS model.
4. **Returns** the generated WAV over HTTP.

---

## 📁 Project Structure

```
voice-cloner-pipeline/
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── cleaner.py      # DSP: resample, noise suppression, trim, normalize
│   │   └── transcriber.py  # ASR: OpenAI Whisper reference transcription
│   ├── models/
│   │   ├── __init__.py
│   │   └── tts_engine.py   # Zero-shot TTS wrapper (F5-TTS / XTTS-v2)
│   └── api/
│       ├── __init__.py
│       └── main.py         # FastAPI endpoints: /api/clone, /health
├── notebooks/
│   └── colab_server.ipynb  # Colab GPU runner with pyngrok tunnel
├── ui/
│   └── app.py              # Streamlit testing UI
├── docker/
│   └── Dockerfile
├── .gitignore
├── requirements.txt
├── main.py                 # Local entry point for the FastAPI server
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.9–3.11** (recommended for PyTorch + whisper compatibility)
- (Optional) NVIDIA GPU with CUDA for accelerated inference

### Local setup

```bash
cd voice-cloner-pipeline

# 1. Create & activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **F5-TTS note**: if the PyPI `f5-tts` package is unavailable, install from
> source:
> ```bash
> pip install git+https://github.com/SWivid/F5-TTS.git
> ```

---

## 🚀 Usage

### 1. Start the FastAPI backend

```bash
python main.py --reload
```

Or directly with Uvicorn:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify: <http://localhost:8000/health>

Interactive docs: <http://localhost:8000/docs>

### 2. Launch the Streamlit UI

In a second terminal:

```bash
streamlit run ui/app.py
```

- Set the **API Endpoint URL** to `http://localhost:8000/api/clone` (default).
- Upload or record 15–30s of reference speech.
- Type the target text and click **Generate Cloned Voice**.
- Play / download the returned WAV.

### 3. (Optional) Docker

```bash
docker build -f docker/Dockerfile -t voice-cloner .
docker run --gpus all -p 8000:8000 voice-cloner
```

### 4. (Optional) Colab GPU + ngrok

Open `notebooks/colab_server.ipynb` on a **GPU runtime** and run all cells.
The notebook installs dependencies, starts uvicorn on port 8000, and exposes
it via **pyngrok**. Paste the printed ngrok URL into the Streamlit sidebar.

---

## 🔌 API Reference

### `POST /api/clone`

Multipart form data:

| Field         | Type        | Description                          |
|---------------|-------------|--------------------------------------|
| `ref_audio`   | File        | Reference audio (WAV/MP3/M4A/FLAC…). |
| `target_text` | Form string | Text to synthesize in the cloned voice. |

**Response**: `audio/wav` — the generated cloned voice clip.

### `GET /health`

Liveness probe returning backend & device info.

```json
{
  "status": "ok",
  "service": "zero-shot-voice-cloner",
  "backend": "f5-tts",
  "device": "cuda"
}
```

---

## 🧠 Architecture

| Layer   | Module                 | Responsibility                                                        |
|---------|------------------------|-----------------------------------------------------------------------|
| DSP     | `src/audio/cleaner.py` | Mono downmix, resample to 24kHz, spectral-gating denoise, trim, peak normalize. |
| ASR     | `src/audio/transcriber.py` | Whisper `base` transcription of the cleaned reference → `ref_text`.   |
| TTS     | `src/models/tts_engine.py` | Zero-shot synthesis (F5-TTS preferred, XTTS-v2 fallback, `mock` for CPU testing). |
| API     | `src/api/main.py`      | FastAPI + CORS, orchestrates clean → transcribe → synthesize → WAV.    |
| UI      | `ui/app.py`            | Streamlit test harness (upload/record + text + player).               |

### TTS backend fallback

`VoiceClonerEngine` auto-detects installed backends:

1. **`f5-tts`** — preferred modern zero-shot TTS.
2. **`xtts`** — Coqui XTTS-v2 (multilingual).
3. **`mock`** — generates a placeholder sine-wave WAV so the full pipeline can
   be exercised end-to-end on machines without a TTS model installed.

---

## 🧪 Testing the pipeline without a GPU

Without a GPU or TTS model, the API still works end-to-end using the **mock**
backend:

```bash
curl -X POST http://localhost:8000/api/clone \
  -F "ref_audio=@sample.wav" \
  -F "target_text=Hello, this is a test of the voice cloning pipeline." \
  -o cloned_voice.wav
```

---

## 📝 Notes

- The first request downloads the Whisper and TTS model weights (can take
  several minutes) — subsequent calls are fast.
- Use clean, single-speaker reference audio for the best cloning quality.
- Models are loaded lazily and cached as process-level singletons to avoid
  re-loading on every request.

## 📄 License

MIT — free to use and modify.

