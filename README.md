# Zero-Shot Voice Cloning System (MATLAB)

This project is a MATLAB Online implementation of a DSP-first zero-shot voice cloning pipeline. The local repository contains text-based MATLAB code only; audio samples and model weights are downloaded or supplied in MATLAB Online.

## Requirements

- MATLAB Online
- Audio Toolbox
- Deep Learning Toolbox
- Signal Processing Toolbox
- A compatible zero-shot TTS model exported as ONNX

## Layout

- `src/dsp/preprocess_signal.m`: mono conversion, resampling to 24 kHz, spectral gating, and peak normalization.
- `src/dsp/stft_analysis.m`: Hann-windowed STFT and explicit triangular Mel filter bank.
- `src/models/load_onnx_engine.m`: ONNX import boundary.
- `src/models/synthesize_features.m`: model inference adapter for text and speaker features.
- `src/vocoder/reconstruct_waveform.m`: iterative ISTFT reconstruction baseline.
- `ui/VoiceClonerApp.m`: programmatic App Designer-compatible UI entry point.
- `scripts/download_weights.m`: MATLAB Online model download utility.

## MATLAB Online quick start

1. Pull the repository into MATLAB Drive.
2. Add the source folders to the MATLAB path:

   ```matlab
   addpath(genpath(pwd));
   ```

3. Load and preprocess a reference clip:

   ```matlab
   [x, fs] = audioread("reference.wav");
   [x_clean, fs] = preprocess_signal(x, fs);
   [mel_matrix, S, f, t] = stft_analysis(x_clean, fs);
   ```

4. Launch the UI:

   ```matlab
   app = VoiceClonerApp;
   ```

5. Download model weights in MATLAB Online with the source URL supplied by the model provider:

   ```matlab
   download_weights("https://example.invalid/model.onnx");
   ```

The ONNX input tensor names and shapes are model-specific. Update `synthesize_features.m` to match the selected export before enabling inference. Model weights, audio files, and MATLAB binary artifacts are excluded by `.gitignore`.

## Validation

MATLAB execution is intentionally not performed on the local machine. Run the quick-start commands and model-specific inference checks in MATLAB Online.
