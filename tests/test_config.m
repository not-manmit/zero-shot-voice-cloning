function test_config()
%TEST_CONFIG Static verification of pipeline configuration parameters and constants.

fprintf("[test_config] Verifying pipeline configuration ...\n");

cfg = pipeline_config();

% Sample rates
assert(cfg.fs == 16000, "cfg.fs must be 16000");
assert(cfg.target_fs == 16000, "cfg.target_fs must be 16000");
assert(cfg.hifigan_sr == 16000, "cfg.hifigan_sr must be 16000");
assert(cfg.spk_sample_rate == 16000, "cfg.spk_sample_rate must be 16000");

% Mel features
assert(cfg.n_mels == 80, "cfg.n_mels must be 80");
assert(cfg.num_mel_bins == 80, "cfg.num_mel_bins must be 80");
assert(cfg.reduction_factor == 2, "cfg.reduction_factor must be 2");

% Tokenizer constants
assert(cfg.vocab_size == 81, "cfg.vocab_size must be 81");
assert(cfg.eos_token_id == 2, "cfg.eos_token_id must be 2");
assert(cfg.pad_token_id == 1, "cfg.pad_token_id must be 1");
assert(cfg.bos_token_id == 0, "cfg.bos_token_id must be 0");
assert(cfg.unk_token_id == 3, "cfg.unk_token_id must be 3");

% Architecture hidden dimensions
assert(cfg.encoder_hidden == 768, "cfg.encoder_hidden must be 768");
assert(cfg.decoder_hidden == 768, "cfg.decoder_hidden must be 768");
assert(cfg.spk_emb_dim == 512, "cfg.spk_emb_dim must be 512");

% Canonical model paths
assert(endsWith(cfg.paths.encoder, "encoder_model.onnx"), "Encoder path incorrect");
assert(endsWith(cfg.paths.decoder, "decoder_model.onnx"), "Decoder path incorrect");
assert(endsWith(cfg.paths.decoder_kv, "decoder_with_past_model.onnx"), "Decoder KV path incorrect");
assert(endsWith(cfg.paths.vocoder, "vocoder_model.onnx"), "Vocoder path incorrect");
assert(endsWith(cfg.paths.spk_encoder, "xvector_encoder.onnx"), "Speaker encoder path incorrect");

fprintf("[test_config] Configuration assertions PASSED.\n");
end
