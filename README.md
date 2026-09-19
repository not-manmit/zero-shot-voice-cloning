# Zero-Shot Voice Cloning in MATLAB

This repository implements a MATLAB-only zero-shot voice cloning pipeline based on a SpeechT5 text-to-speech backbone and a learned x-vector speaker encoder. The project is structured for MATLAB Online execution and avoids Python-based inference or local model conversion.

## Final architecture

Reference audio
  -> preprocessing
  -> x-vector speaker encoder
  -> speaker embedding
  -> text tokenizer
  -> SpeechT5 encoder
  -> SpeechT5 autoregressive decoder
  -> acoustic/Mel features
  -> neural vocoder
  -> waveform
  -> playback and WAV export

## Required MATLAB products

- MATLAB
- Signal Processing Toolbox
- Audio Toolbox
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox
- Parallel Computing Toolbox (optional, only if ONNX runtime benefits from it)

## Canonical model layout

All model files must live under the repository root and use the same layout throughout the codebase:

```text
models/
├── speecht5/
│   ├── encoder_model.onnx
│   ├── decoder_model.onnx
│   ├── decoder_with_past_model.onnx
│   ├── vocoder_model.onnx
│   └── tokenizer_vocab.json   (optional, downloaded if available)
└── xvector/
    ├── xvector_encoder.onnx
    └── cmu_arctic_xvectors.mat
```

## MATLAB Online workflow

1. Open MATLAB Online.
2. Clone or pull the repository.
3. Change into the project folder:

   ```matlab
   cd("zero-shot-voice-cloning-main")
   ```

4. Run the setup script:

   ```matlab
   run("scripts/setup_matlab_online.m")
   ```

5. Validate model compatibility:

   ```matlab
   validate_models
   ```

6. Launch the app:

   ```matlab
   VoiceClonerApp
   ```

7. Record or upload a reference voice, enter target text, and click Generate.
8. Play or save the output waveform with `sound()` or `audiowrite()`.
9. Run the test suite:

   ```matlab
   run("tests/test_tokenizer.m")
   run("tests/test_speaker_encoder.m")
   run("tests/test_model_loading.m")
   run("tests/test_vocoder.m")
   run("tests/test_end_to_end.m")
   ```

## Repository layout

```text
.
├── CONTEXT.md
├── README.md
├── WORKFLOW.md
├── docs/
│   └── ARCHITECTURE.md
├── models/
│   ├── speecht5/
│   └── xvector/
├── scripts/
│   ├── download_weights.m
│   └── setup_matlab_online.m
├── src/
│   ├── config/
│   │   └── pipeline_config.m
│   ├── dsp/
│   │   ├── preprocess_signal.m
│   │   └── stft_analysis.m
│   ├── inference/
│   │   └── generate_voice.m
│   ├── models/
│   │   ├── load_onnx_engine.m
│   │   ├── synthesize_features.m
│   │   └── validate_models.m
│   ├── speaker/
│   │   └── extract_speaker_embedding.m
│   ├── text/
│   │   └── tokenize_text.m
│   ├── utils/
│   │   └── check_requirements.m
│   └── vocoder/
│       └── reconstruct_waveform.m
├── tests/
│   ├── test_tokenizer.m
│   ├── test_speaker_encoder.m
│   ├── test_model_loading.m
│   ├── test_vocoder.m
│   └── test_end_to_end.m
└── ui/
    └── VoiceClonerApp.m
```

## Model download and setup

The repository does not store large ONNX model weights in Git. The setup script checks and downloads only missing assets into the canonical model directories. This is executed inside MATLAB Online, not on the developer laptop.

## Important limitation

This repository is intentionally written to use MATLAB-native ONNX import and runtime APIs. The exact model contract is validated at runtime by inspecting the imported network. If a particular ONNX export is incompatible with the local MATLAB version or unsupported operators, the validation function reports the exact failure rather than pretending the model works.

## Notes

- The final system uses 16 kHz as the SpeechT5 model rate.
- The pipeline is designed to avoid repeated model reloads and to reuse cached ONNX runtime objects across generations.
- No local Python environment, conversion scripts, or external inference server is required.
