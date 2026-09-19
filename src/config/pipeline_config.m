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
cfg.encoder_hidden = 768;
cfg.decoder_hidden = 768;
cfg.num_mel_bins = 80;
cfg.reduction_factor = 2;

cfg.max_text_len = 450;
cfg.min_len_ratio = 0.0;
cfg.max_len_ratio = 20.0;
cfg.stop_threshold = 0.5;

cfg.spk_emb_dim = 512;
cfg.spk_min_dur_s = 1.0;
cfg.spk_chunk_dur_s = 5.0;
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
cfg.paths.spk_embeddings = fullfile(model_root, 'xvector', 'cmu_arctic_xvectors.mat');
cfg.paths.tokenizer_vocab = fullfile(model_root, 'speecht5', 'tokenizer_vocab.json');

cfg.onnx = struct();
cfg.onnx.encoder_inputs = {'input_ids', 'attention_mask'};
cfg.onnx.encoder_outputs = {'last_hidden_state'};
cfg.onnx.decoder_inputs = {'input_ids', 'encoder_hidden_states', 'encoder_attention_mask', 'speaker_embeddings'};
cfg.onnx.decoder_outputs = {'output', 'logits', 'past_key_values'};
cfg.onnx.vocoder_inputs = {'spectrogram'};
cfg.onnx.vocoder_outputs = {'waveform'};
cfg.onnx.spk_inputs = {'input'};
cfg.onnx.spk_outputs = {'output'};

cfg.samples_per_frame = cfg.hop_length;

cfg.required_toolboxes = {'Audio Toolbox', 'Deep Learning Toolbox', 'Signal Processing Toolbox'};
end
