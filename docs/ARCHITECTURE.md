# Zero-Shot Voice Cloning Architecture (MATLAB-Only)

## Diagram (actual implementation — verified contracts)

```
Reference WAV (any fs, stereo/mono)
  | audioread
  v
preprocess_signal  -- mono mean, resample 16 kHz (polyphase anti-alias, rat), DC remove, STFT Hann noise gate (first 0.5 s), peak norm to [-1,1]
  | x_clean [N,1] @16kHz, N=1024 hop=256
  v
extract_speaker_embedding -- CAM++ fbank80: 25ms win 10ms hop 80 mels 20-8000Hz + log + CMN -> ONNX openspeech voxceleb_CAM++.onnx [B,T,80]->[B,512] per chunk (3s/50% overlap, average, L2). Explicit input 'features', no heuristic fallback.
  | spk_emb [1,512] L2=1  (genuinely conditions decoder every step via speaker_embeddings)
  |
Text string -- tokenize_text: lowercase, whitespace collapse, 81-vocab char map (VOCAB 42 surface + 39 reserved), BOS 0/PAD 1/EOS 2/unk 3, attention_mask [1,T], trunc 450
  | input_ids int64 [1,T], attention_mask int64 [1,T]
  v
SpeechT5 encoder (Xenova encoder_model.onnx)  [1,T]->[1,T,768]  Explicit inputs: input_ids, attention_mask
  | encoder_hidden_states float32 [1,T,768]
  v
SpeechT5 decoder  (decoder_model.onnx first step, decoder_with_past_model.onnx loop)
  inputs each step (explicit names): input_ids int64 [1,1] (=BOS), encoder_hidden_states [1,T,768], encoder_attention_mask [1,T], speaker_embeddings [1,512], past_key_values.* float32 (only in with_past)
  outputs: logits [1,1,80 (+optional stop dim 81)], past_key_values.* per layer
  loop: explicit KV output->input positional map; until stop logit sigmoid>0.5 or max_decoder_steps (500) or max_len_ratio*text_len (20x), with finite/energy guards; max 500 frames
  | acoustic Mel [80,T_mel] (natural log, floor 1e-10, 80 bins 80-7600 Hz)
  v
HiFi-GAN (Xenova speecht5_hifigan model.onnx)  spectrogram float32 [1,80,T_mel] (explicit 'spectrogram') -> waveform [1,T*256] @16kHz (256x upsample)
  | peak norm clip [-1,1]
  v
generate_voice result: waveform [N,1] double 16kHz, sampleRate, speakerEmbedding [1,512], acousticFeatures [80,T], tokenIds/mask, metrics (pre/spk/tok/encDec/voc/total), metadata
  |
UI VoiceClonerApp: reference plot, mel imagesc [80 x T], generated plot, Play (sound), Save (audiowrite), Status trace 5 stages, Metrics label, embedding cache
```

## Contracts (from src/models/model_contract.m — cross-checked with pipeline_config.m:onnx)

| Model | File | Inputs (exact name dtype shape) | Outputs | Notes |
|---|---|---|---|---|
| Encoder | speecht5/encoder_model.onnx | input_ids int64 [B,T], attention_mask int64 [B,T] | last_hidden_state float32 [B,T,768] | B=1, T dynamic, opset 14 |
| Decoder | speecht5/decoder_model.onnx | input_ids int64 [1,1], encoder_hidden_states float32 [B,T,768], encoder_attention_mask int64 [B,T], speaker_embeddings float32 [B,512] | logits float32 [B,1,80/81], past_key_values float32 per layer | first step only, BOS=0 |
| Decoder_with_past | speecht5/decoder_with_past_model.onnx | same + past_key_values.* float32 [B,heads,prev_len,head_dim] | logits float32 [B,1,80/81], past_key_values.* | loop, encoder states + speaker constant; order via output->input positional map |
| Vocoder | speecht5/vocoder_model.onnx (Xenova model.onnx) | spectrogram float32 [B,80,T] (natural log Mel) | waveform float32 [B,1,T*256] | upsample 256x, 16kHz |
| CAM++ | xvector/xvector_encoder.onnx (voxceleb_CAM++.onnx) | features float32 [B,T,80] (fbank+CMN) | embedding float32 [B,512] pre-norm | chunked 3s 50% overlap, L2 normalise |

Validate with `diagnose_onnx` and `validate_models` in MATLAB Online — they print InputNames/OutputNames verbatim. No `contains("mask")` heuristic survives in production path.

## DSP chain details

- Resample: `rat` + `resample` polyphase anti-alias (Nyquist–Shannon).
- Noise gate: STFT Hann 1024 hop 256, per-bin threshold median(noise_seg 0.5s)*1.5, gain clamped [0.08,1], istft reconstruct.
- Mel: power |STFT|^2 -> 80 triangular HTK filters 2595*log10(1+f/700), f_min 80 f_max 7600, log(max(mel,1e-10)) natural log.
- Speaker fbank: independent 512pt FFT Hann 400 hop160, 80 mels 20-8000 Hz, log + CMN (time-mean zero) matching Wespeaker preprocessing.

## Caching

`load_onnx_engine` persistent `CACHED_MODELS` loaded once per session (`force_reload` to clear). UI `SpeakerEmbedding` cached via `ReferenceHash` (numel+sum cheap hash) — reused if reference unchanged. No model reload on Generate.

## Failure mode

`validate_models` distinguishes `exists` vs `ok` (importable + contract name match + output count). `synthesize_features`/`extract_speaker_embedding`/`reconstruct_waveform` use explicit name checks (`strcmpi`) and throw `MExxx:ContractMismatch/UnexpectedInput/InferenceFailed` on violation — never silently send arbitrary tensor. Tests assert `Required model assets are unavailable` instead of silent pass. No `double(char(text))`, random embeddings, or Griffin-Lim fallback.

## MATLAB Online

`setup_matlab_online` finds project root, adds paths, runs `check_requirements` (Audio, DSP, Deep Learning + ONNX converter, MATLAB R2023b+), creates `models/speecht5`+`models/xvector`, calls `download_weights` (real Xenova/openspeech URLs via `websave`, skips if >1 MB), then `validate_models` with dlnetwork import. Total ~875 MB fp32 (fits 20 GB Drive). Use `diagnose_onnx` for truth export.

## Sample rate consistency

Only 16 kHz everywhere: `pipeline_config.fs = target_fs = hifigan_sr = spk_sample_rate = 16000`, `hop 256 (16ms) win 1024 (64ms)`, speaker `25ms/10ms`. No 24 kHz path.

## Generate path trace

`VoiceClonerApp.generateSpeech` -> `generate_voice` -> `preprocess_signal` (16k mono denoise) -> `extract_speaker_embedding` (CAM++ fbank+CMN+ONNX avg+L2) -> `tokenize_text` (81 vocab BOS/EOS) -> `synthesize_features` (encoder predict + decoder loop with past KV + speaker condition every step -> [80,T]) -> `reconstruct_waveform` (HiFi-GAN [1,80,T]->waveform) -> `result.waveform` -> `sound`/`audiowrite`/`imagesc` in UI. Every arrow points to real code; `decoder_with_past_model.onnx` actually used after step 1 via explicit KV map.
