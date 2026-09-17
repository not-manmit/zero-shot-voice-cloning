function cfg = pipeline_config()
%PIPELINE_CONFIG Central configuration for the SpeechT5 + HiFi-GAN pipeline.
%
%   cfg = PIPELINE_CONFIG() returns a scalar struct that carries every
%   constant required by the voice-cloning pipeline.  All magic numbers
%   live here; every other function reads from this struct instead of
%   embedding literals.
%
%   Architecture: SpeechT5 (Microsoft) zero-shot TTS
%                + SpeechBrain x-vector speaker encoder
%                + HiFi-GAN neural vocoder
%
%   Reference:
%     Ao et al. 2021 – "SpeechT5: Unified-Modal Encoder-Decoder Pre-
%     Training for Spoken Language Processing"
%     https://arxiv.org/abs/2110.07205
%
%   All model ONNX files are downloaded by scripts/download_weights.m.
%   Do NOT commit model files to Git.

% -----------------------------------------------------------------------
% Audio signal conventions
% -----------------------------------------------------------------------
cfg.fs               = 16000;   % sample rate expected by all SpeechT5 models (Hz)
cfg.target_fs        = 16000;   % alias – used by preprocess_signal

% -----------------------------------------------------------------------
% Mel-spectrogram feature extractor (SpeechT5FeatureExtractor defaults)
% -----------------------------------------------------------------------
cfg.n_fft            = 1024;    % DFT length (samples)  [N in N-point DFT]
cfg.hop_length_ms    = 16;      % frame step  (ms) → 256 samples @ 16 kHz
cfg.win_length_ms    = 64;      % analysis window (ms) → 1024 samples @ 16 kHz
cfg.hop_length       = round(cfg.hop_length_ms  * cfg.fs / 1000);  % 256
cfg.win_length       = round(cfg.win_length_ms  * cfg.fs / 1000);  % 1024
cfg.n_mels           = 80;      % number of Mel filter-bank bands
cfg.f_min            = 80;      % lower Mel edge (Hz)
cfg.f_max            = 7600;    % upper Mel edge (Hz)
cfg.mel_floor        = 1e-10;   % clamp before log  (prevents -Inf)
cfg.mel_scale        = 'htk';   % HTK formula  (not Slaney)

% -----------------------------------------------------------------------
% SpeechT5 text encoder / decoder
% -----------------------------------------------------------------------
cfg.vocab_size       = 81;      % number of tokens (incl. <pad>,<s>,</s>,<unk>)
cfg.pad_token_id     = 1;       % <pad>
cfg.bos_token_id     = 0;       % <s>   (decoder start)
cfg.eos_token_id     = 2;       % </s>  (decoder stop)
cfg.encoder_hidden   = 768;     % encoder hidden-state dimension
cfg.decoder_hidden   = 768;     % decoder hidden-state dimension
cfg.num_mel_bins     = 80;      % mel frames output per decoder step
cfg.reduction_factor = 2;       % mel frames per decoder step (TTS post-net)

% -----------------------------------------------------------------------
% Autoregressive decoding limits
% -----------------------------------------------------------------------
cfg.max_text_len     = 450;     % maximum tokenised text length
cfg.min_len_ratio    = 0.0;     % minimum output length  ratio  (vs. input)
cfg.max_len_ratio    = 20.0;    % maximum output length  ratio  (vs. input)
cfg.stop_threshold   = 0.5;     % sigmoid probability threshold for EOS token

% -----------------------------------------------------------------------
% Speaker encoder (SpeechBrain TDNN x-vector)
% -----------------------------------------------------------------------
cfg.spk_emb_dim      = 512;     % x-vector embedding dimension
cfg.spk_min_dur_s    = 1.0;     % minimum reference duration (seconds)
cfg.spk_chunk_dur_s  = 5.0;     % chunk length for multi-segment averaging

% -----------------------------------------------------------------------
% HiFi-GAN vocoder
% -----------------------------------------------------------------------
cfg.hifigan_sr       = 16000;   % vocoder output sample rate (Hz)

% -----------------------------------------------------------------------
% Model paths  (relative to repository root, populated at download time)
% -----------------------------------------------------------------------
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % repo root
model_root = fullfile(root, 'models');

cfg.model_root   = model_root;

cfg.paths.encoder          = fullfile(model_root, 'speecht5',    'encoder_model.onnx');
cfg.paths.decoder          = fullfile(model_root, 'speecht5',    'decoder_model.onnx');
cfg.paths.decoder_kv       = fullfile(model_root, 'speecht5',    'decoder_with_past_model.onnx');
cfg.paths.vocoder          = fullfile(model_root, 'speecht5',    'vocoder_model.onnx');
cfg.paths.spk_encoder      = fullfile(model_root, 'xvector',     'xvector_encoder.onnx');
cfg.paths.spk_embeddings   = fullfile(model_root, 'xvector',     'cmu_arctic_xvectors.mat');

% -----------------------------------------------------------------------
% ONNX tensor contracts  (names as seen after importONNXNetwork)
%
%   NOTE: importONNXNetwork may rename inputs by replacing special chars
%   with underscores and prepending 'x_'.  The names below are the ONNX
%   graph names before import.  The model-loading code inspects the
%   actual loaded network and caches the resolved names.
% -----------------------------------------------------------------------
cfg.onnx.encoder_inputs   = {'input_ids', 'attention_mask'};
cfg.onnx.encoder_outputs  = {'last_hidden_state'};

cfg.onnx.decoder_inputs   = {'input_ids', 'encoder_hidden_states', ...
                              'encoder_attention_mask', 'speaker_embeddings'};
cfg.onnx.decoder_outputs  = {'output', 'logits', 'past_key_values'};

cfg.onnx.vocoder_inputs   = {'spectrogram'};
cfg.onnx.vocoder_outputs  = {'waveform'};

cfg.onnx.spk_inputs       = {'input'};
cfg.onnx.spk_outputs      = {'output'};

% -----------------------------------------------------------------------
% Derived convenience constants
% -----------------------------------------------------------------------
cfg.samples_per_frame = cfg.hop_length;   % alias used by vocoder length calc

end
