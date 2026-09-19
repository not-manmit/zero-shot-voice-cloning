# Zero-Shot Voice Cloning Architecture

## Final pipeline

Reference audio
  -> mono + resample to 16 kHz
  -> denoise / normalize
  -> x-vector speaker encoder
  -> speaker embedding
  -> text tokenizer
  -> SpeechT5 encoder
  -> SpeechT5 autoregressive decoder
  -> acoustic feature frames
  -> neural vocoder
  -> waveform
  -> playback / save

## Stage-by-stage contract

### 1. Reference audio preprocessing
- Input: waveform array and original sample rate
- Output: mono 16 kHz waveform
- MATLAB API: `audioread`, `resample`, `stft`, `istft`
- Normalization: DC removal, noise suppression, peak normalization

### 2. Speaker encoder
- Input: preprocessed reference waveform
- Output: learned x-vector embedding
- Model: `models/xvector/xvector_encoder.onnx`
- Expected output dimension: 512
- Normalization: L2-normalize to unit length

### 3. Text tokenization
- Input: target text string
- Output: integer token IDs and attention mask
- Contract: deterministic vocabulary mapping with BOS/EOS handling
- Normalization: lowercase, whitespace collapse, unknowns mapped to `<unk>`

### 4. SpeechT5 text encoder
- Input: token IDs and attention mask
- Output: encoder hidden states
- Model: `models/speecht5/encoder_model.onnx`
- Expected hidden size: 768

### 5. SpeechT5 decoder
- Inputs: token IDs, encoder states, attention mask, speaker conditioning
- Output: acoustic feature frames
- Model: `models/speecht5/decoder_model.onnx` and optional `decoder_with_past_model.onnx`
- Handles: first pass and cached autoregressive pass if the export requires it

### 6. Neural vocoder
- Input: acoustic feature frames
- Output: waveform
- Model: `models/speecht5/vocoder_model.onnx`
- Output rate: 16 kHz
- MATLAB API: `importONNXNetwork` followed by `predict`

## MATLAB Online suitability

This design is appropriate for MATLAB Online because it: 
- uses MATLAB-native Audio Toolbox and Deep Learning Toolbox APIs
- keeps model loading persistent in a session cache
- avoids local Python or training workflows
- downloads only the required ONNX assets into the project storage area
- does not require GPU conversion or cloud inference servers

## Required verification step

The exact input/output contract for each ONNX export must be validated in MATLAB Online before generation. This repository checks the imported graph and reports unsupported operators or missing assets explicitly rather than pretending a model is usable.
