# Zero-Shot Voice Cloning in MATLAB (MATLAB-Only Architecture)

A MATLAB-only zero-shot acoustic feature synthesis and voice-cloning pipeline built using **SpeechT5** (Xenova ONNX exports of `microsoft/speecht5_tts`), the **CAM++ speaker embedding model** (`openspeech/wespeaker-models`), and the **HiFi-GAN neural vocoder** (`Xenova/speecht5_hifigan`).

**Execution Target:** Designed for execution directly inside **MATLAB Online** (with MATLAB Drive storage). Zero Python runtime dependencies: no PyTorch, no Conda, no Docker, no external web servers.

---

## Pipeline Execution Graph

```text
Reference Speech Audio (any sampling rate, mono or stereo)
        ↓
1. Nyquist–Shannon Resampling (16 kHz anti-aliasing) + DC Removal + Peak Normalization
        ↓
2. Acoustic Filterbank Extraction: 80-bin log-Mel + Cepstral Mean Normalization (CMN)
        ↓
3. CAM++ ONNX Speaker Encoder (CBT layout: [1, T, 80] -> [1, 512] L2-normalized embedding)
        ↓
4. Target Text Tokenization: SentencePiece 81-token char vocabulary (metaspace ▁, EOS only)
        ↓
5. SpeechT5 Encoder Forward Pass: [1, T] -> [1, T, 768] contextual hidden representations
        ↓
6. SpeechT5 Decoder Step 1: zero Mel frame [1, 1, 80] -> initial Mel [2, 80] + initial KV cache
        ↓
7. Autoregressive Loop (with past KV): feed last Mel frame [1, 1, 80] + KV cache (rf=2)
        ↓
8. 80-bin Mel Spectrogram [80 x T] (natural log scale, 80–7600 Hz)
        ↓
9. HiFi-GAN Neural Vocoder (SCB layout: [1, 80, T] -> [1, 1, T*256] 16 kHz discrete waveform)
        ↓
10. Waveform Normalization & Playback / Audio File Export (.wav)
```

---

## Canonical Model Inventory

All weights are downloaded from verified public Hugging Face repositories into the canonical `models/` folder:

| Sub-Model | Canonical File Path | Source Repository | Expected Size | Opset |
|---|---|---|---|---|
| Text Encoder | `models/speecht5/encoder_model.onnx` | Xenova/speecht5_tts | ~343 MB | 14 (fp32) |
| Decoder (Step 1) | `models/speecht5/decoder_model.onnx` | Xenova/speecht5_tts | ~238 MB | 14 (fp32) |
| Decoder (KV Loop) | `models/speecht5/decoder_with_past_model.onnx` | Xenova/speecht5_tts | ~210 MB | 14 (fp32) |
| HiFi-GAN Vocoder | `models/speecht5/vocoder_model.onnx` | Xenova/speecht5_hifigan | ~55.4 MB | 14 (fp32) |
| Speaker Encoder | `models/xvector/xvector_encoder.onnx` | openspeech/wespeaker-models | ~29.3 MB | 14 (fp32) |

**Total Model Data:** ~875.7 MB (well within MATLAB Drive's 20 GB standard quota).  
*Note: Model binaries are excluded from Git version control via `.gitignore`.*

---

## Verification Status Legend

To adhere strictly to source-of-truth integrity, contracts and implementation components are categorized as follows:

- **[IMPLEMENTED & STATICALLY VERIFIED]:** Written in MATLAB, unit-tested without models (tokenization, DSP math, tensor utilities, configuration).
- **[OBSERVED AT RUNTIME]:** Confirmed directly from the imported ONNX graph inside MATLAB Online.
- **[PENDING ONLINE INSPECTION]:** Contract expected based on Hugging Face / Transformers.js architectures; requires running `scripts/diagnose_onnx.m` in MATLAB Online.

---

## Complete MATLAB Online Execution Guide

Follow this step-by-step sequence in **MATLAB Online**:

### Step 1: Open Repository in MATLAB Online
Upload or clone the repository into your MATLAB Drive workspace, then navigate into the directory:
```matlab
cd("zero-shot")   % or the name of your cloned folder
```

### Step 2: Run Automated Environment Setup
Executes path configuration, toolbox checks, model file acquisition via `websave`, and initial validation:
```matlab
run("scripts/setup_matlab_online.m")
```

### Step 3: Run Master Preflight Validation
Verifies all toolboxes, model file integrity, contract compatibility, and component smoke tests:
```matlab
run("scripts/validate_matlab_online.m")
```
*Confirm that `MATLAB ONLINE PREFLIGHT: PASS` is displayed.*

### Step 4: Run Forensic Contract Diagnostics (Optional / Troubleshooting)
If any model contract fails or requires inspection, run the forensic analyzer to view exact tensor names and shapes:
```matlab
run("scripts/diagnose_onnx.m")
```

### Step 5: Run Individual Component Tests
Run individual unit and smoke tests as needed:
```matlab
run("tests/test_config.m")             % Static config test
run("tests/test_tokenizer.m")          % Static tokenization test
run("tests/test_dsp.m")                % Static DSP & STFT test
run("tests/test_model_loading.m")      % Model loading test
run("tests/test_encoder.m")            % Text encoder inference test
run("tests/test_speaker_encoder.m")    % Speaker encoder structural test
run("tests/test_decoder.m")            % First-step & KV cache decoder test
run("tests/test_vocoder.m")            % Vocoder synthesis test
run("tests/test_end_to_end.m")         % Synthetic pipeline integrity smoke test
```

### Step 6: Validate Against Real Human Speech
To validate voice cloning on real speech:
1. Place a 3–5 second clean `.wav` speech file in `tests/reference_speech.wav` (or set `setenv('TEST_REFERENCE_AUDIO', 'path/to/speech.wav')`).
2. Run the test:
```matlab
run("tests/test_end_to_end_real_reference.m")
```

### Step 7: Launch Interactive Voice Cloning App
Launch the interactive App Designer interface:
```matlab
VoiceClonerApp
```
1. Click **Upload WAV** to select a reference speech file (or use **Record** if microphone permissions are active in your browser).
2. Enter target English synthesis text in the text area.
3. Click **Generate** — the app displays genuine stage progress, accurate measured timings, waveform plots, and Mel spectrograms.
4. Click **Play Output** or **Save WAV** to export the cloned speech.

---

## Signal Processing Details (Signals & Systems Focus)

- **Resampling:** Anti-aliasing polyphase rational filter (`resample`, `rat`) to convert input rates to 16 kHz.
- **DC Bias Removal:** Subtraction of the arithmetic mean $y[n] = x[n] - \mu_x$.
- **Windowing:** Periodic Hann windowing ($N = 1024$, hop = 256 samples, 75% frame overlap).
- **HTK Mel Filterbank:** 80 triangular filters spanning 80 Hz to 7600 Hz.
- **Speaker Feature Extraction:** 80-bin filterbank with 25 ms window, 10 ms frame shift, and utterance-level Cepstral Mean Normalization (CMN).
- **Speaker Conditioning:** CAM++ 512-dim embedding, L2-normalized ($\|e\|_2 = 1.0$), passed to the decoder at each autoregressive step.
- **Decoupled Embedding Caching:** When reference speech is unchanged, the 512-dim embedding is genuinely reused, skipping repeated extraction.

---

## Known Constraints & Operational Limits

- **MATLAB Version:** Requires MATLAB R2023b or R2024a+ with the *Deep Learning Toolbox Converter for ONNX Model Format* add-on.
- **SpeechT5 Model Scope:** SpeechT5 base is trained on 16 kHz English speech (LibriTTS). Prosody and quality depend on clean, uncorrupted reference speech.
- **Inference Latency:** CPU inference in MATLAB Online takes approximately 8–20 seconds per synthesized sentence.
- **Microphone in MATLAB Online:** Web browser security policies may restrict direct `audiorecorder` access in some environments; WAV file upload serves as the primary deterministic path.
