# Zero-Shot Voice Cloning Architecture (MATLAB-Only) — Verified 2026-05-11

## Diagram (verified: HF config + tokenizer.json + modeling_speecht5.py + transformers.js)

```
Reference WAV (any fs, stereo/mono)
  | audioread
  v
preprocess_signal  -- mono mean, resample 16 kHz (polyphase anti-alias, rat), DC remove, STFT Hann 1024 hop256 noise gate (first 0.5 s median*1.5 gain [0.08,1] istft), peak norm to [-1,1]
  | x_clean [N,1] @16kHz, N=1024 hop=256
  v
extract_speaker_embedding -- CAM++ fbank80: 25ms win 10ms hop 80 mels 20-8000Hz + log + CMN -> ONNX openspeech voxceleb_CAM++.onnx [1,T,80] CBT float32 'features' ->[B,512] per chunk (3s/50% overlap, average, L2). Single layout, no fallback.
  | spk_emb [1,512] L2=1  (conditions decoder prenet every step via speaker_embeddings)
  |
Text string -- tokenize_text: SentencePiece-char (tokenizer.json 81 vocab, ▁ U+2581 metaspace, WhitespaceSplit+Metaspace add_prefix_space, Split), case PRESERVED, no lower, no BOS prepend, EOS 2 appended only, attention_mask [1,T], trunc 450
  | input_ids int64 [1,T], attention_mask int64 [1,T]  (T = tokens +1 EOS)
  v
SpeechT5 encoder (Xenova encoder_model.onnx)  [1,T]->[1,T,768]  Explicit inputs: input_ids int64, attention_mask int64
  | encoder_hidden_states float32 [1,T,768]  + encoder_attention_mask int64 [1,T]
  v
SpeechT5 decoder  (decoder_model.onnx first step, decoder_with_past_model.onnx loop, or merged with use_cache_branch bool)
  inputs each step VERIFIED: output_sequence float32 [1,1,80] zeros start then last predicted Mel frame, encoder_hidden_states [1,T,768], encoder_attention_mask [1,T], speaker_embeddings [1,512], past_key_values.* float32 (only with_past), [use_cache_branch bool]
  outputs VERIFIED: spectrum float32 [1,rf,80] (feat_out), prob float32 [1,rf] (prob_out sigmoid), past_key_values.* per layer (rf=2 reduction_factor per config.json)
  loop: zero Mel start -> prenet+speak concat -> wrapped_decoder last frame + KV -> spectrum[rf,80] prob[rf] -> cat spectrum frames -> sigmoid(prob)>0.5 stop per frame after minSteps (min_len_ratio) or max_decoder_steps (500//rf) or max_len_ratio*text_len/rf; no energy guard
  | acoustic Mel [80,T_mel] natural log, floor 1e-10, 80 bins 80-7600 Hz (stft_analysis) vs speaker 20-8000 (dual)
  v
HiFi-GAN (Xenova speecht5_hifigan model.onnx)  spectrogram float32 [1,80,T_mel] SCB (explicit 'spectrogram') -> waveform [1,T*256] @16kHz (256x upsample)
  | peak norm clip [-1,1]
  v
generate_voice result: waveform [N,1] double 16kHz, sampleRate, speakerEmbedding [1,512], acousticFeatures [80,T], tokenIds/mask, metrics (pre/spk/tok/enc+dec/voc/total), metadata
  |
UI VoiceClonerApp: reference plot, mel imagesc [80 x T], generated plot, Play (sound), Save (audiowrite), Status trace 5 stages, Metrics label, embedding cache
```

## Contracts (verified vs model_contract.m — observed flag)

| Model | File | Inputs (exact name dtype shape) | Outputs | Notes | Observed |
|---|---|---|---|---|---|
| Encoder | speecht5/encoder_model.onnx | input_ids int64 [B,T], attention_mask int64 [B,T] | last_hidden_state float32 [B,T,768] | B=1, T dynamic, opset 14, no BOS in input (EOS only) | true |
| Decoder | speecht5/decoder_model.onnx | output_sequence float32 [B,1,80] (zeros start), encoder_hidden_states float32 [B,T,768], encoder_attention_mask int64 [B,T], speaker_embeddings float32 [B,512] | spectrum float32 [B,rf,80], prob float32 [B,rf], past_key_values.* float32 per layer | rf=2 per config.json:2, legacy int input_ids BOS fallback detected at runtime | false* |
| Decoder_with_past | speecht5/decoder_with_past_model.onnx | same + past_key_values.* float32 [B,heads,prev_len,head_dim] + optional use_cache_branch bool | spectrum float32 [B,rf,80], prob float32 [B,rf], past_key_values.* | merged model uses bool flag; split pair 238+210 MB downloaded via download_weights | false* |
| Vocoder | speecht5/vocoder_model.onnx (Xenova model.onnx) | spectrogram float32 [B,80,T] natural log | waveform float32 [B,1,T*256] | SCB layout, 256x upsample 16kHz, needs diagnose_onnx shape confirm | false* |
| CAM++ | xvector/xvector_encoder.onnx (voxceleb_CAM++.onnx) | features float32 [B,T,80] CBT [1,T,80] | embedding float32 [B,512] pre-norm | chunked 3s 50% overlap L2, single layout | false* |

*`false` = local ONNX absent (models/ not present, .gitignore excludes *.onnx); contract derived from HF authoritative sources, needs `diagnose_onnx` confirmation on MATLAB Online. Validate with `diagnose_onnx` and `validate_models` — they now print past tensor counts, use_cache_branch, float vs legacy detection.

## DSP chain details

- Resample: `rat` + `resample` polyphase anti-alias (Nyquist–Shannon).
- Noise gate: STFT Hann 1024 hop 256, per-bin threshold median(noise_seg 0.5s)*1.5, gain clamped [0.08,1], istft reconstruct.
- Mel: power |STFT|^2 -> 80 triangular HTK filters 2595*log10(1+f/700), f_min 80 f_max 7600, log(max(mel,1e-10)) natural log. Reduction factor 2 => 2 frames per decoder step, so T_mel = steps*2.
- Speaker fbank: independent 512pt FFT Hann 400 hop160, 80 mels 20-8000 Hz, log + CMN (time-mean zero) matching Wespeaker preprocessing. CMN is time-mean subtraction per utterance.
- Tokenizer: 81 vocab from tokenizer.json:model.vocab (▁=4, e=5, ..., "—" U+2014=70, æ=72, é=73, ê=76, œ=77, ̄=78, <mask>=79, <ctc_blank>=80). WhitespaceSplit + Metaspace(add_prefix_space) + Split, then TemplateProcessing appends </s> (2) only.

## Caching

`load_onnx_engine` persistent `CACHED_MODELS` loaded once per session (`force_reload` to clear). UI `SpeakerEmbedding` cached via `ReferenceHash` (numel+sum cheap hash) — reused if reference unchanged. No model reload on Generate.

## Failure mode

`validate_models` now distinguishes float Mel vs legacy int BOS contract, counts past tensors, checks use_cache_branch, validates spectrum/prob vs packed 81 legacy. `synthesize_features` uses verified float output_sequence contract ([1,1,80] zeros) with rf=2 spectral head + separate prob head (sigmoid), supports legacy fallback with warning. `extract_speaker_embedding` single layout CBT, `reconstruct_waveform` single layout SCB. Tests assert `Required model assets are unavailable` instead of silent pass.

## MATLAB Online

`setup_matlab_online` finds project root, adds paths, runs `check_requirements` (Audio, DSP, Deep Learning + ONNX converter, MATLAB R2023b+), creates `models/speecht5`+`models/xvector`, calls `download_weights` (real Xenova/openspeech URLs via `websave`, skips if >1 MB), then `validate_models` with dlnetwork import. Total ~875 MB fp32 (fits 20 GB Drive). Use `diagnose_onnx` for truth export — now reports file size, opset, per-tensor dtype, past tensor enumeration, float vs legacy detection.

## Sample rate consistency

Only 16 kHz everywhere: `pipeline_config.fs = target_fs = hifigan_sr = spk_sample_rate = 16000`, `hop 256 (16ms) win 1024 (64ms)`, speaker `25ms/10ms` (400/160). `reduction_factor=2` verified per config.json. No 24 kHz path.

## Generate path trace

`VoiceClonerApp.generateSpeech` -> `generate_voice` -> `preprocess_signal` (16k mono denoise) -> `extract_speaker_embedding` (CAM++ fbank+CMN+ONNX avg+L2) -> `tokenize_text` (SentencePiece char ▁ + EOS only, 81 vocab, verified IDs) -> `synthesize_features` (zero Mel [1,1,80] -> encoder predict -> decoder loop rf=2 spectrum+prob with past KV + speaker condition every step -> [80,T]) -> `reconstruct_waveform` (HiFi-GAN [1,80,T]->waveform) -> `result.waveform` -> `sound`/`audiowrite`/`imagesc` in UI. Every arrow points to verified code; `decoder_with_past_model.onnx` actually used after step 1 via explicit KV map + use_cache_branch handling.
