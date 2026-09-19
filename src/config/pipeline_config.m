function cfg = pipeline_config()
%PIPELINE_CONFIG Central configuration for the SpeechT5 + x-vector pipeline.
%
% This project follows a single canonical model layout:
%   models/speecht5/*.onnx
%   models/xvector/*.onnx
%
% SpeechT5 exports are expected to operate at 16 kHz and emit 80-bin
% log-Mel or acoustic features. The actual imported ONNX network is used to
% validate the exact input and output names before inference.

cfg.fs = 16000;
cfg.target_fs = 16000;
cfg.n_fft = 1024;
cfg.hop_length_ms = 16;
cfg.win_length_ms = 64;
cfg.hop_length = round(cfg.hop_length_ms * cfg.fs / 1000);
cfg.win_length = round(cfg.win_length_ms * cfg.fs / 1000);
cfg.n_mels = 80;
cfg.f_min = 80;
cfg.f_max = 7600;
cfg.mel_floor = 1e-10;
cfg.mel_scale = 'htk';

cfg.vocab_size = 81;
cfg.pad_token_id = 1;
cfg.bos_token_id = 0;
cfg.eos_token_id = 2;
cfg.unk_token_id = 3;
cfg.encoder_hidden = 768;
cfg.decoder_hidden = 768;
cfg.num_mel_bins = 80;
cfg.reduction_factor = 1;  % Xenova export uses reduction_factor=1 (no stacking)

cfg.max_text_len = 450;
cfg.min_len_ratio = 0.0;
cfg.max_len_ratio = 20.0;
cfg.stop_threshold = 0.5;
cfg.max_decoder_steps = 500; % autoregressive limit (aligns with SpeechT5 max mel length)
cfg.eos_mel_threshold = 0.5; % stop logit threshold if model emits it

cfg.spk_emb_dim = 512;      % SpeechT5 expects 512; CAM++ export also outputs 512
cfg.spk_min_dur_s = 1.0;
cfg.spk_chunk_dur_s = 3.0;  % CAM++/Wespeaker uses 3 s chunks with 50% overlap
cfg.spk_sample_rate = 16000;
cfg.fbank_n_mels = 80;      % speaker front-end uses 80-dim log-Mel with CMN
cfg.hifigan_sr = 16000;

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
model_root = fullfile(root, 'models');

cfg.model_root = model_root;
cfg.paths = struct();
cfg.paths.speecht5_root = fullfile(model_root, 'speecht5');
cfg.paths.xvector_root = fullfile(model_root, 'xvector');
cfg.paths.encoder = fullfile(model_root, 'speecht5', 'encoder_model.onnx');
cfg.paths.decoder = fullfile(model_root, 'speecht5', 'decoder_model.onnx');
cfg.paths.decoder_kv = fullfile(model_root, 'speecht5', 'decoder_with_past_model.onnx');
cfg.paths.vocoder = fullfile(model_root, 'speecht5', 'vocoder_model.onnx');
cfg.paths.spk_encoder = fullfile(model_root, 'xvector', 'xvector_encoder.onnx');
cfg.paths.spk_embeddings = fullfile(model_root, 'xvector', 'cmu_arctic_xvectors.mat'); % legacy, not required
cfg.paths.tokenizer_vocab = fullfile(model_root, 'speecht5', 'tokenizer_vocab.json'); % optional

% Explicit ONNX contracts verified against Xenova / openspeech exports via netx-ray.
%  - Xenova/speecht5_tts encoder_model.onnx expects int64 input_ids [B,T] + attention_mask [B,T] -> last_hidden_state [B,T,768]
%  - decoder_model.onnx expects input_ids [B,1] (start token), encoder_hidden_states [B,T,768], attention_mask [B,T], speaker_embeddings [B,512] -> logits [B,1,80] + past_key_values
%  - decoder_with_past_model.onnx expects same plus past_key_values.* -> logits [B,1,80] + updated past
%  - Xenova/speecht5_hifigan model.onnx expects spectrogram [B,80,T] or [1,80,T] -> waveform [1,1,T*256]
%  - openspeech CAM++ expects fbank [B,T,80] -> embedding [B,512]
cfg.onnx = struct();
cfg.onnx.encoder_inputs = {'input_ids', 'attention_mask'};
cfg.onnx.encoder_outputs = {'last_hidden_state'};
cfg.onnx.decoder_inputs = {'input_ids', 'encoder_hidden_states', 'encoder_attention_mask', 'speaker_embeddings'};
cfg.onnx.decoder_outputs = {'logits', 'past_key_values'};
cfg.onnx.decoder_with_past_inputs = {'input_ids', 'encoder_hidden_states', 'encoder_attention_mask', 'speaker_embeddings', 'past_key_values'};
cfg.onnx.decoder_with_past_outputs = {'logits', 'past_key_values'};
cfg.onnx.vocoder_inputs = {'spectrogram'};
cfg.onnx.vocoder_outputs = {'waveform'};
cfg.onnx.spk_inputs = {'features'}; % [B, T, 80] fbank
cfg.onnx.spk_outputs = {'embedding'};

cfg.samples_per_frame = cfg.hop_length;

cfg.required_toolboxes = {'Audio Toolbox', 'Deep Learning Toolbox', 'Signal Processing Toolbox'};
end
