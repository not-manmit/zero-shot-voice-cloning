# Zero-Shot Voice Cloning System

A MATLAB application for generating speech in a speaker's voice from a short reference recording and target text. The system combines digital signal processing, speaker representation, ONNX-based model inference, and waveform reconstruction in a modular pipeline.

## Features

- Load a WAV reference recording.
- Convert audio to mono and resample it to 24 kHz.
- Reduce stationary background noise and normalize signal amplitude.
- Generate Hann-windowed STFT and Mel-spectrogram features.
- Import a compatible zero-shot text-to-speech model through ONNX.
- Reconstruct an audio waveform from synthesized acoustic features.
- View the reference waveform and Mel-spectrogram in the application UI.

## Requirements

- MATLAB Online or desktop MATLAB
- Audio Toolbox
- Deep Learning Toolbox
- Signal Processing Toolbox
- A compatible zero-shot TTS model exported in ONNX format

## Project Structure

```text
.
├── scripts/
│   └── download_weights.m
├── src/
│   ├── dsp/
│   │   ├── preprocess_signal.m
│   │   └── stft_analysis.m
│   ├── models/
│   │   ├── load_onnx_engine.m
│   │   └── synthesize_features.m
│   └── vocoder/
│       └── reconstruct_waveform.m
└── ui/
    └── VoiceClonerApp.m
```

## Quick Start

1. Open the project in MATLAB.
2. Add the project folders to the MATLAB path:

   ```matlab
   addpath(genpath(pwd));
   ```

3. Launch the application:

   ```matlab
   app = VoiceClonerApp;
   ```

4. Use **Load reference** to select a WAV file.
5. Enter the sentence to synthesize.
6. Select **Synthesize** to preprocess the reference and generate its acoustic features.

## Processing Pipeline

```text
Reference WAV
     │
     ▼
Preprocessing
     │  mono, resampling, noise suppression, normalization
     ▼
STFT and Mel Features ──► Speaker Representation
                              │
Target Text ────────────────┐  │
                            ▼  ▼
                         ONNX TTS Model
                              │
                              ▼
                       Waveform Reconstruction
                              │
                              ▼
                         Synthesized Speech
```

The main reusable functions are:

- `preprocess_signal`: prepares the reference signal at 24 kHz.
- `stft_analysis`: computes the STFT and Mel-spectrogram matrix.
- `load_onnx_engine`: imports the selected ONNX network.
- `synthesize_features`: adapts target text and speaker data to the model.
- `reconstruct_waveform`: produces a time-domain waveform from acoustic features.

## Model Weights

Model weights are not stored in the repository. Download them into the MATLAB Drive environment with:

```matlab
download_weights("https://your-model-host.example/model.onnx", "tts_model.onnx");
```

The model should be placed at:

```text
models/weights/tts_model.onnx
```

The ONNX input names, tensor shapes, tokenizer, and speaker-embedding format depend on the selected model. Update `synthesize_features.m` to match the model export before running inference.

## Example DSP Usage

```matlab
[x, fs] = audioread("reference.wav");
[x_clean, fs] = preprocess_signal(x, fs);
[mel_matrix, S, f, t] = stft_analysis(x_clean, fs);

plot(t, x_clean);
xlabel("Time (s)");
ylabel("Amplitude");
title("Preprocessed Reference Audio");
```

Audio files, model weights, generated outputs, and MATLAB binary artifacts are excluded from version control.
