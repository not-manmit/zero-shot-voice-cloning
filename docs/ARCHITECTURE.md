# Zero-Shot Voice Cloning System Architecture (MATLAB Online Edition)

## System Overview

This system is an **Acoustic Feature Synthesis and Neural Vocoding Engine** written strictly in **MATLAB**, designed for execution in **MATLAB Online**. It synthesizes target speech from arbitrary text while matching the vocal characteristics of an acoustic reference utterance (zero-shot voice cloning).

---

## Detailed Block Diagram

```text
+---------------------------------------------------------------------------------------+
|                               REFERENCE SPEECH PIPELINE                                |
+---------------------------------------------------------------------------------------+
                                  Reference WAV Speech
                                            │
                                            ▼
                        preprocess_signal.m (Signals & Systems DSP)
                        - Multi-channel spatial downmix to mono
                        - Anti-aliasing rational polyphase resampling to 16 kHz
                        - Zero-frequency DC offset elimination: y[n] = x[n] - E[x]
                        - Optional conservative energy-guided spectral gating
                        - Peak normalization to [-1.0, 1.0]
                                            │
                                            ▼
                        compute_campp_fbank80 (Speaker Front-End)
                        - STFT (Hann window 400 samples, hop 160 samples, N_fft=512)
                        - 80-bin triangular Mel filterbank (20 Hz - 8000 Hz)
                        - Logarithmic compression: log(max(power, 1e-10))
                        - Cepstral Mean Normalization (CMN) per segment
                                            │
                                            ▼
                        models/xvector/xvector_encoder.onnx (CAM++)
                        - Input: 'features' [1, T, 80] dlarray (CBT format)
                        - Output: 'embedding' [1, 512] float32
                        - L2-normalization: spk_emb = emb / ||emb||_2
                                            │
                        [Genuine Cache in VoiceClonerApp if ref unchanged]
                                            │
                                            ▼
                                   spk_emb [1, 512]

+---------------------------------------------------------------------------------------+
|                               TARGET TEXT & TTS ENCODER                               |
+---------------------------------------------------------------------------------------+
                                   Target Text String
                                            │
                                            ▼
                        tokenize_text.m (SentencePiece 81-Token Vocabulary)
                        - Whitespace split & metaspace prefix "▁" (U+2581)
                        - Exact character-level mapping to vocab IDs [0..80]
                        - Case preserved (e.g., 'H'=35, 'h'=11)
                        - Special token handling: EOS (id 2) appended; no BOS prepended
                                            │
                                            ▼
                        models/speecht5/encoder_model.onnx
                        - Inputs: input_ids [1, T], attention_mask [1, T] ('CB' int64)
                        - Output: last_hidden_state [1, T, 768] ('CB' / float32)

+---------------------------------------------------------------------------------------+
|                       AUTOREGRESSIVE DECODER & KV CACHE                               |
+---------------------------------------------------------------------------------------+
        enc_hidden [1, T, 768] + enc_mask [1, T] + spk_emb [1, 512]
                                            │
               Step 1: output_sequence = zero-Mel frame [1, 1, 80]
                                            │
                                            ▼
                        models/speecht5/decoder_model.onnx
                        - Inputs: output_sequence [1,1,80], enc_hidden, enc_mask, spk_emb
                        - Outputs:
                          * spectrum [2, 80] (reduction_factor = 2)
                          * prob [1, 2] (stop logits / probabilities)
                          * past_key_values.* (initial KV cache tensors)
                                            │
                 ┌──────────────────────────┴──────────────────────────┐
                 │                                                     │
                 ▼                                                     ▼
    Accumulate Mel Frames                             Are stop criteria satisfied?
    acoustic_frames = [acoustic_frames; spectrum]     - sigmoid(prob) > 0.5 (after minSteps)
                 │                                    - stagnation / energy collapse
                 │                                    - step >= max_decoder_steps
                 ▼                                                     │
    Prepare next input:                                          Yes: Break
    output_sequence = last Mel frame [1, 1, 80]                  No:  Continue
                 │                                                     │
                 ▼                                                     │
    models/speecht5/decoder_with_past_model.onnx                       │
    - Feed: output_sequence, enc_hidden, enc_mask,                     │
            spk_emb, past_key_values.*, [use_cache_branch]             │
    - Updates KV cache and emits next 2 Mel frames                     │
                 │                                                     │
                 └────────────────────── Loop ─────────────────────────┘
                                            │
                                            ▼
                        Acoustic Mel Spectrogram [80 x T_mel]
                        (Natural log scale, HTK 80-7600 Hz, floor 1e-10)

+---------------------------------------------------------------------------------------+
|                                NEURAL VOCODER & WAV                                   |
+---------------------------------------------------------------------------------------+
                               Acoustic Mel [80 x T_mel]
                                            │
                                            ▼
                        models/speecht5/vocoder_model.onnx (HiFi-GAN)
                        - Input: 'spectrogram' [1, 80, T_mel] dlarray ('SCB' float32)
                        - Output: 'waveform' [1, 1, T_mel * 256] (256x upsampling)
                                            │
                                            ▼
                        reconstruct_waveform.m
                        - Normalization to unity amplitude: max(|x[n]|) = 1.0
                        - Hard clipping to [-1.0, 1.0]
                                            │
                                            ▼
                        16 kHz Mono Waveform Sequence [N x 1]
                        - Interactive UI playback (sound)
                        - Persistent disk export (audiowrite)
```

---

## Exact Tensor Specifications & Layout Contracts

| Component | Tensor Name | Semantic Role | Dimension | MATLAB dlarray Format | Data Type | Verification Status |
|---|---|---|---|---|---|---|
| **Encoder** | `input_ids` | Token sequence (with EOS) | `[1, T]` | `'CB'` | `int64` | **[OBSERVED]** |
| | `attention_mask` | Binary sequence mask | `[1, T]` | `'CB'` | `int64` | **[OBSERVED]** |
| | `last_hidden_state` | Contextual representations | `[1, T, 768]` | `'CB'` / `[1,T,768]` | `float32` | **[OBSERVED]** |
| **Speaker** | `features` | 80-bin filterbank with CMN | `[1, T, 80]` | `'CBT'` | `float32` | **[OBSERVED]** |
| | `embedding` | Speaker embedding vector | `[1, 512]` | Matrix `[1, 512]` | `float32` | **[OBSERVED]** |
| **Decoder** | `output_sequence` | Previous Mel frame(s) | `[1, 1, 80]` | `'CBT'` or `'SCB'` | `float32` | **[OBSERVED]** |
| | `encoder_hidden_states` | Cross-attention context | `[1, T, 768]` | `'CB'` | `float32` | **[OBSERVED]** |
| | `encoder_attention_mask`| Cross-attention mask | `[1, T]` | `'CB'` | `int64` | **[OBSERVED]** |
| | `speaker_embeddings` | Speaker conditioning | `[1, 512]` | `'CB'` | `float32` | **[OBSERVED]** |
| | `spectrum` | Predicted Mel frames | `[rf, 80]` | Matrix `[2, 80]` | `float32` | **[OBSERVED]** |
| | `prob` | Stop logit / probability | `[1, rf]` | Row `[1, 2]` | `float32` | **[OBSERVED]** |
| | `past_key_values.*` | Self & cross-attention cache | Layer-dependent | dlarray cell array | `float32` | **[OBSERVED]** |
| **Vocoder** | `spectrogram` | Log-Mel acoustic features | `[1, 80, T]` | `'SCB'` | `float32` | **[OBSERVED]** |
| | `waveform` | 16 kHz discrete waveform | `[1, 1, T*256]` | Vector `[N, 1]` | `float32` | **[OBSERVED]** |

---

## Autoregressive Reduction Factor ($rf = 2$)

In standard SpeechT5 (`microsoft/speecht5_tts/config.json`), the decoder features a reduction factor of $rf = 2$.
This means each step of the decoder generates $2$ consecutive Mel frames rather than a single frame:
$$\text{Total Mel Frames } T_{\text{mel}} = \text{Steps} \times 2$$
With a hop size of $256$ samples at $16\text{ kHz}$ ($16\text{ ms}$ per frame), each decoder step synthesizes $32\text{ ms}$ of continuous speech.

---

## Calibrated Stop-Probability Semantics

The decoder produces raw output logits from the stop prediction head. If values fall outside $[0, 1]$, the pipeline computes:
$$P(\text{stop}) = \frac{1}{1 + e^{-z}}$$
A stop decision is triggered when:
1. Total accumulated frames $> \text{minSteps} \times rf$
2. Calibrated $P(\text{stop}) > \text{cfg.stop\_threshold}$ (default: $0.5$)
3. Or temporal energy across the last 5 frames falls below $10^{-7}$ (numerical silence guard)
4. Or total steps reach `cfg.max_decoder_steps` ($500$, yielding up to $1000$ Mel frames $\approx 16\text{ s}$).

---

## Architectural Isolation & MATLAB-Only Enforcement

- All deep learning models execute natively via MATLAB's `dlnetwork` engine (`importNetworkFromONNX` or `importONNXNetwork`).
- No Python binaries, wrappers, or inter-process communication layers exist anywhere in the runtime.
- The pipeline adheres strictly to academic Signals & Systems terminology and methodologies.
