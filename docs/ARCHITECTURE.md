# Zero-Shot Voice Cloning Architecture (MATLAB-Only)

## Diagram (actual implementation)

```
Reference WAV (any fs, stereo/mono)
  | audioread
  v
preprocess_signal  -- mono mean, resample 16 kHz (polyphase), DC remove, STFT noise gate (first 0.5 s), peak norm to [-1,1]
  | x_clean [N,1] @16kHz
  v
extract_speaker_embedding -- CAM++ fbank80: 25ms win 10ms hop 80 mels 20-8000Hz + log + CMN -> ONNX openspeech voxceleb_CAM++.onnx [B,T,80]->[B,512] per chunk (3s/50% overlap, average, L2)
  | spk_emb [1,512] L2=1  (genuinely conditions decoder every step)
  |
Text string -- tokenize_text: lowercase, whitespace collapse, 81-vocab char map, BOS 0/EOS 2/unk 3, attention_mask, trunc 450
  | input_ids [1,T], attention_mask [1,T]
  v
SpeechT5 encoder (Xenova encoder_model.onnx)  [1,T]->[1,T,768]
  | encoder_hidden_states
  v
SpeechT5 decoder  (decoder_model.onnx first step, decoder_with_past_model.onnx loop)
  inputs each step: input_ids [1,1], encoder_hidden_states [1,T,768], encoder_attention_mask [1,T], speaker_embeddings [1,512], past_key_values.*
  outputs: logits [1,1,80 (+optional stop)], past_key_values.*
  loop: until stop logit >0.5 or max_decoder_steps (500) or max_len_ratio*text_len, with finite/energy guards
  | acoustic Mel [80,T_mel] (natural log, floor 1e-10, 80 bins)
  v
HiFi-GAN (Xenova speecht5_hifigan model.onnx)  [1,80,T_mel] -> waveform [T_mel*256] @16kHz
  | peak norm, clip
  v
generate_voice result: waveform, sampleRate, speakerEmbedding, acousticFeatures, tokenIds/mask, metrics, metadata
  |
UI VoiceClonerApp: reference plot, mel imagesc [80 x T], generated plot, Play (sound), Save (audiowrite)
```

## Contracts (from src/models/model_contract.m)

| Model | File | Inputs (name dtype shape) | Outputs | Notes |
|---|---|---|---|---|
| Encoder | speecht5/encoder_model.onnx | input_ids int64 [B,T], attention_mask int64 [B,T] | last_hidden_state float32 [B,T,768] | B=1, T dynamic |
| Decoder | speecht5/decoder_model.onnx | input_ids int64 [1,1], encoder_hidden_states float32 [B,T,768], encoder_attention_mask int64 [B,T], speaker_embeddings float32 [B,512] | logits float32 [B,1,80/81], past_key_values float32 | first step only |
| Decoder_with_past | speecht5/decoder_with_past_model.onnx | same + past_key_values.* float32 [B,heads,prev,dim] | logits float32 [B,1,80/81], past_key_values.* | loop, keep encoder states constant |
| Vocoder | speecht5/vocoder_model.onnx (Xenova model.onnx) | spectrogram float32 [B,80,T] | waveform float32 [B,1,T*256] | upsample 256x, 16kHz, Mel is log |
| CAM++ | xvector/xvector_encoder.onnx (voxceleb_CAM++.onnx) | features float32 [B,T,80] (fbank+CMN) | embedding float32 [B,512] pre-norm | chunked 3s, L2 normalise |

See `model_contract.m` for programmatic access; `validate_models.m` checks that imported graphs actually expose exactly these names.

## DSP chain details

- Resample: `rat` + `resample` polyphase anti-alias.
- Noise gate: STFT Hann 1024 hop 256, per-bin threshold median(noise) *1.5, gain clamped [0.08,1].
- Mel: triangular HTK filters 2595*log10(1+f/700), power->mel->log(mel_floor).
- Speaker fbank: separate function with CMN (zero-mean over time) matching Wespeaker preprocessing.

## Caching

`load_onnx_engine` uses a persistent `CACHED_MODELS` struct loaded once per MATLAB session. UI keeps `ReferenceHash` and reuses `SpeakerEmbedding` if reference unchanged. No model reload on every Generate.

## Failure mode

`validate_models` distinguishes `exists` vs `ok` (importable + contract match). `synthesize_features` does not fallback to `double(char(text))`, random embeddings, or Griffin-Lim – every step throws an explicit `MExxx` error if contract violated. Tests use `assert` and report `Required model assets are unavailable` instead of silent pass.

## MATLAB Online

`setup_matlab_online` finds project root, adds paths, runs `check_requirements` (Audio, DSP, Deep Learning + ONNX converter), creates dirs, calls `download_weights` (real Xenova/openspeech URLs via `websave`), then `validate_models`. `download_weights` skips present files after 1 MB sanity check. Total download ~875 MB.

## Sample rate consistency

Only 16 kHz. `pipeline_config.fs = target_fs = hifigan_sr = spk_sample_rate = 16000`. No 24 kHz code path exists.

## Generate path trace

`VoiceClonerApp.generateSpeech` -> `generate_voice` -> `preprocess_signal` -> `extract_speaker_embedding` (CAM++ fbank + ONNX + avg+L2) -> `tokenize_text` -> `synthesize_features` (encoder + decoder loop with past KV + speaker condition) -> `reconstruct_waveform` (HiFi-GAN) -> `result.waveform` -> `sound`/`audiowrite`/`imagesc` in UI. Every arrow points to real code; `decoder_with_past_model.onnx` is actually used inside the loop.
