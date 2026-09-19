# Zero-Shot Voice Cloning in MATLAB (MATLAB-Only, No Python at Runtime)

MATLAB-only zero-shot TTS using **SpeechT5** (Xenova ONNX exports of `microsoft/speecht5_tts`) + **CAM++ speaker encoder** (`openspeech/wespeaker-models`) + **HiFi-GAN** (`Xenova/speecht5_hifigan`). No PyTorch, no Conda, no local Python needed at inference – all execution happens in **MATLAB Online**.

Pipeline:

```
Ref audio (any rate) -> preprocess 16kHz mono denoise -> CAM++ fbank80 + CMN -> 512-dim L2 x-vector
                                                              |
Text -> SpeechT5 tokenizer (81 vocab, BOS/EOS) -> SpeechT5 encoder (768) -> autoregressive decoder + past KV (80-bin Mel)
                                                                                              |
                                                                                         Mel [80,T]
                                                                                              |
                                                                                         HiFi-GAN -> waveform 16kHz -> play / save WAV
```

## Exact models (real URLs, verified on Hugging Face)

| File (canonical) | Purpose | Source repo | URL | Size |
|---|---|---|---|---|
| `models/speecht5/encoder_model.onnx` | SpeechT5 encoder | Xenova/speecht5_tts | https://huggingface.co/Xenova/speecht5_tts/resolve/main/onnx/encoder_model.onnx | 343 MB |
| `models/speecht5/decoder_model.onnx` | SpeechT5 decoder step 1 | Xenova/speecht5_tts | https://huggingface.co/Xenova/speecht5_tts/resolve/main/onnx/decoder_model.onnx | 238 MB |
| `models/speecht5/decoder_with_past_model.onnx` | SpeechT5 decoder with KV cache | Xenova/speecht5_tts | https://huggingface.co/Xenova/speecht5_tts/resolve/main/onnx/decoder_with_past_model.onnx | 210 MB |
| `models/speecht5/vocoder_model.onnx` | HiFi-GAN | Xenova/speecht5_hifigan | https://huggingface.co/Xenova/speecht5_hifigan/resolve/main/onnx/model.onnx | 55.4 MB |
| `models/xvector/xvector_encoder.onnx` | CAM++ 512-dim speaker encoder | openspeech/wespeaker-models | https://huggingface.co/openspeech/wespeaker-models/resolve/main/voxceleb_CAM++.onnx | 29.3 MB |

Total ~875 MB (fits in 20 GB MATLAB Drive). Quantized variants (`*_quantized.onnx`, `*_int8.onnx`) from same repos can be used if storage-constrained – rename to same canonical names.

Licenses: MIT (SpeechT5/HiFi-GAN) + Apache-2.0/CC-BY-4.0 (CAM++, Xenova tooling). Check each repo before redistribution.

## Canonical layout (must match across all code)

```
models/
  speecht5/
    encoder_model.onnx
    decoder_model.onnx
    decoder_with_past_model.onnx
    vocoder_model.onnx
  xvector/
    xvector_encoder.onnx
```

`pipeline_config.m` defines all paths; `model_contract.m` defines exact input/output names/shapes; `download_weights.m` downloads to exactly these paths.

## MATLAB requirements (used, not invented)

- MATLAB R2023b+ minimum, R2024a+ recommended (ONNX opset 14 support; quantized opset 17 needs R2024a)
- Audio Toolbox (audioread/write, sound, audiorecorder)
- Signal Processing Toolbox (stft, istft, resample, hann)
- Deep Learning Toolbox (+ `dlarray`, `dlnetwork`, `predict`)
- **Deep Learning Toolbox Converter for ONNX Model Format** (for `importONNXNetwork`) — install via Add-Ons > Get Add-Ons > “ONNX”
- Parallel Computing Toolbox – optional (GPU not required; CPU inference ~5-20 s/sentence)

Check with: `check_requirements`  – reports missing toolboxes + MATLAB version. Also run `diagnose_onnx` to print actual ONNX InputNames/OutputNames.

## MATLAB Online – one-shot setup

```matlab
% 1. Clone (or upload) and cd
cd("zero-shot-voice-cloning-main")  % name may vary on Drive

% 2. Setup (paths + dirs + download missing weights + validation)
run("scripts/setup_matlab_online.m")

% 3. Validate contracts in detail
validate_models
diagnose_onnx   % optional: exact ONNX InputNames/OutputNames dump

% 4. Launch UI
VoiceClonerApp
```

`setup_matlab_online.m` will:
- add `src`, `ui`, `scripts` to path
- create `models/speecht5` and `models/xvector`
- download only missing ONNX from the real URLs above (via `websave`)
- call `validate_models` which imports each ONNX and checks input/output names

If `validate_models` reports `ok=false`, do not click Generate – see the error (e.g. missing Support Package, opset too new).

## App usage

1. **Reference voice**: Record (Record → Stop) or Upload WAV (any rate, automatically resampled 16 kHz). Must be >=1 s and non-silent; 3-5 s recommended.
2. **Target text**: English, lowercased internally, 450 token limit (truncates with warning).
3. **Generate**: Runs `generate_voice` which shows tokenizing → encoder → autoregressive decoder (with past KV) → vocoder. Reuses the 512-dim speaker embedding if reference unchanged.
4. **Play / Save**: Play via `sound`, Save via `audiowrite` (16 kHz WAV).

`generate_voice` returns: `waveform [N,1]`, `sampleRate`, `speakerEmbedding [1,512]`, `acousticFeatures [80,T]`, `tokenIds`, `attentionMask`, `metrics` (pre/spk/tok/enc+dec/voc/total seconds), `metadata`.

## Tests (use MATLAB unittest assertions)

```matlab
run("tests/test_tokenizer.m")
run("tests/test_speaker_encoder.m")   % needs models present
run("tests/test_model_loading.m")     % validates all ONNX contracts
run("tests/test_vocoder.m")
run("tests/test_end_to_end.m")       % ref (synthetic) -> embedding -> TTS -> vocoder -> wav
```

If `Required model assets are unavailable` is printed, the test correctly reports missing assets rather than silently passing.

## Sample rate

Everything is 16 kHz. `pipeline_config.fs = target_fs = hifigan_sr = 16000`. `stft_analysis` uses N=1024, hop=256 (16 ms), win=1024 (64 ms), 80 mels 80-7600 Hz, natural log. CAM++ front-end uses 25 ms/10 ms, 80 mels 20-8000 Hz, CMN. No 24 kHz exists anywhere.

## Limitations (honest)

- Xenova SpeechT5 ONNX is an export of the base LibriTTS checkpoint – prosody limited; long sentences may degrade after ~500 Mel frames (max_decoder_steps).
- CAM++ generalises to unseen speakers but is not finetuned on SpeechT5 speaker space; embedding is L2-projected and genuinely conditions the decoder (checked at every step).
- MATLAB `importONNXNetwork` supports opset <=17; the fp32 Xenova exports are opset 14 – compatible with R2023b+. Quantized exports (8-bit) require newer MATLAB and may not import.
- No training, no fine-tuning, no GPU required.
- All inference is CPU in MATLAB Online; expect ~5-20 s per sentence depending on length.

## Docs

- `docs/ARCHITECTURE.md` – contract details
- `src/models/model_contract.m` – machine-readable contract
- `src/config/pipeline_config.m` – sample rate, mel, token, model paths
