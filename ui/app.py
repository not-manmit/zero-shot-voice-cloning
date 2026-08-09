"""
Streamlit UI - Zero-Shot Voice Cloning Test Harness.

Provides a clean interface to exercise the FastAPI backend:

    * Sidebar: API endpoint URL (local or ngrok).
    * Reference audio: file upload OR in-browser recorder (30s max).
    * Target text: text area for the utterance to synthesize.
    * "Generate Cloned Voice" button -> POST multipart to /api/clone.
    * Audio player to play the returned cloned WAV.

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import io

import requests
import streamlit as st

# ----------------------------- Page config ------------------------------- #

st.set_page_config(
    page_title="Zero-Shot Voice Cloning Engine",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ Sidebar ---------------------------------- #

with st.sidebar:
    st.header("⚙️ Configuration")
    default_endpoint = "http://localhost:8000/api/clone"
    api_endpoint = st.text_input(
        "API Endpoint URL",
        value=default_endpoint,
        help="Local FastAPI URL or an ngrok URL from Colab.",
    ).strip().rstrip("/")

    st.divider()
    st.caption("Pipeline: DSP Cleaner → Whisper ASR → F5-TTS/XTTS-v2")

    if st.button("🔍 Health Check", use_container_width=True):
        health_url = api_endpoint.replace("/api/clone", "/health")
        try:
            resp = requests.get(health_url, timeout=5)
            st.success(resp.json())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Health check failed: {exc}")

# ------------------------------- Header ---------------------------------- #

st.title("🎙️ Zero-Shot Voice Cloning Engine (DSP + GenAI)")
st.caption(
    "Upload 15–30s of clear reference speech, type the target text, and "
    "generate a cloned voice sample."
)

# --------------------------- Reference audio ------------------------------ #

col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Reference Audio")
    source = st.radio(
        "Audio source",
        ["Upload a file", "Record audio"],
        horizontal=True,
        help="Use 15–30 seconds of clean, single-speaker speech.",
    )

    ref_audio_bytes: bytes | None = None

    if source == "Upload a file":
        uploaded = st.file_uploader(
            "Choose a reference clip",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            help="Supported: WAV, MP3, M4A, FLAC, OGG.",
        )
        if uploaded is not None:
            ref_audio_bytes = uploaded.getvalue()
            st.audio(ref_audio_bytes, format=uploaded.type)
    else:
        recorder_audio = st.audio_input(
            "Record 15–30 seconds of speech",
            help="Speak clearly; the clip is used to clone the voice.",
        )
        if recorder_audio is not None:
            ref_audio_bytes = recorder_audio.getvalue()
            st.audio(ref_audio_bytes, format="audio/wav")

with col2:
    st.subheader("2️⃣ Target Text")
    target_text = st.text_area(
        "Text to synthesize in the cloned voice",
        placeholder="Type a sentence the cloned voice should speak...",
        height=180,
    )
    st.caption(f"Characters: {len(target_text)}")

# ------------------------------ Generate --------------------------------- #

st.divider()

if st.button("🚀 Generate Cloned Voice", type="primary", use_container_width=True):
    if ref_audio_bytes is None:
        st.error("Please provide reference audio (upload or record).")
        st.stop()
    if not target_text.strip():
        st.error("Please enter target text to synthesize.")
        st.stop()

    files = {"ref_audio": ("reference.wav", ref_audio_bytes, "audio/wav")}
    data = {"target_text": target_text}

    with st.spinner("Cloning voice... this may take a while on first run."):
        try:
            resp = requests.post(api_endpoint, files=files, data=data, timeout=600)
        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not reach {api_endpoint}. Is the FastAPI server running?"
            )
            st.stop()

    if resp.status_code == 200:
        st.success("✅ Voice cloned successfully!")
        st.audio(resp.content, format="audio/wav")

        st.download_button(
            label="💾 Download cloned voice (.wav)",
            data=resp.content,
            file_name="cloned_voice.wav",
            mime="audio/wav",
        )
    else:
        detail = resp.text
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            pass
        st.error(f"Cloning failed (HTTP {resp.status_code}): {detail}")

st.divider()
st.caption(
    "Note: First request downloads & loads the TTS + Whisper models, which "
    "can take several minutes. Subsequent requests are faster."
)

