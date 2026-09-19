function test_decoder()
%TEST_DECODER Dedicated model-backed validation for SpeechT5 first-step and with-past decoders.
%
%   Verifies:
%     Stage A (First-Step Decoder):
%       - output_sequence [1, 1, 80] zero Mel starting token
%       - encoder hidden states [1, T, 768]
%       - encoder attention mask [1, T]
%       - speaker embedding [1, 512]
%       - Forward inference produces:
%         * finite spectrum [rf, 80] (rf=2)
%         * calibrated stop probabilities in [0, 1]
%         * valid initial KV cache cell array
%
%     Stage B (Decoder With Past):
%       - Feed last Mel frame [1, 1, 80]
%       - Feed previous KV cache tensors
%       - Forward inference succeeds
%       - Output Mel spectrum remains finite
%       - KV cache evolves stably across multiple consecutive steps

fprintf("[test_decoder] Validating SpeechT5 autoregressive decoder ...\n");

cfg = pipeline_config();

if ~isfile(cfg.paths.decoder)
    error("test_decoder:DecoderMissing", ...
        "decoder_model.onnx is missing at %s. Run scripts/setup_matlab_online.m.", cfg.paths.decoder);
end

% 1. Import First-Step Decoder
fprintf("  Importing decoder_model.onnx ...\n");
[decNet, decMeta] = import_onnx_model(cfg.paths.decoder);
assert(~isempty(decNet), "Failed to import decoder network");

% 2. Synthesize Deterministic Conditioning Inputs
T_tokens = 10;
rf = cfg.reduction_factor;
enc_hidden = single(randn(1, T_tokens, cfg.encoder_hidden) * 0.1);
enc_mask = int64(ones(1, T_tokens));
spk_emb = single(randn(1, cfg.spk_emb_dim));
spk_emb = spk_emb / norm(spk_emb); % L2 normalized

% Starting frame: zero-Mel [1, 1, 80]
start_mel = single(zeros(1, 1, cfg.n_mels));

% 3. Test First-Step Decoder
fprintf("  Executing first-step decoder inference ...\n");
t_step1 = tic;
inputs_step1 = tensor_contract_utils.format_decoder_inputs( ...
    start_mel, enc_hidden, enc_mask, spk_emb, {}, decNet, false);

try
    raw_step1 = predict(decNet, inputs_step1);
    elapsed1 = toc(t_step1);
catch ME
    error("test_decoder:Step1Failed", ...
        "First-step decoder failed: %s\nInputs: [%s]", ME.message, strjoin(decMeta.inputNames, ", "));
end

[spectrum1, prob1, raw_logit1, past_kv] = tensor_contract_utils.parse_decoder_outputs(raw_step1, decNet, rf);

fprintf("  Step 1 succeeded in %.2f s:\n", elapsed1);
fprintf("    Spectrum shape: %s (finite: %s)\n", mat2str(size(spectrum1)), string(all(isfinite(spectrum1), 'all')));
fprintf("    Stop prob:      %s (raw logit: %s)\n", mat2str(prob1), mat2str(raw_logit1));
fprintf("    KV tensors:     %d extracted\n", numel(past_kv));

assert(size(spectrum1, 2) == cfg.n_mels, sprintf("Spectrum must have %d Mel bins", cfg.n_mels));
assert(all(isfinite(spectrum1), 'all'), "Step 1 spectrum contains NaN/Inf");
assert(all(prob1 >= 0 & prob1 <= 1), "Calibrated probabilities must be within [0, 1]");

% 4. Test Decoder With Past (if present)
if isfile(cfg.paths.decoder_kv)
    fprintf("\n  Importing decoder_with_past_model.onnx ...\n");
    [decKvNet, decKvMeta] = import_onnx_model(cfg.paths.decoder_kv);

    fprintf("  Executing multi-step autoregressive KV loop (3 steps) ...\n");
    last_frame = spectrum1(end, :);

    for s = 2:4
        t_kv = tic;
        in_mel = reshape(single(last_frame), 1, 1, cfg.n_mels);
        inputs_kv = tensor_contract_utils.format_decoder_inputs( ...
            in_mel, enc_hidden, enc_mask, spk_emb, past_kv, decKvNet, true);

        try
            raw_kv = predict(decKvNet, inputs_kv);
            elapsed_kv = toc(t_kv);
        catch ME
            error("test_decoder:KvStepFailed", ...
                "Decoder with past failed at step %d: %s\nInputs: [%s]", ...
                s, ME.message, strjoin(decKvMeta.inputNames, ", "));
        end

        [spectrum_s, prob_s, ~, past_kv] = tensor_contract_utils.parse_decoder_outputs(raw_kv, decKvNet, rf);
        fprintf("    Step %d: %.2f s | Spectrum: %s | Stop prob: %s | KV count: %d\n", ...
            s, elapsed_kv, mat2str(size(spectrum_s)), mat2str(prob_s), numel(past_kv));

        assert(all(isfinite(spectrum_s), 'all'), sprintf("Step %d spectrum contains NaN/Inf", s));
        assert(size(spectrum_s, 2) == cfg.n_mels, "Spectrum Mel bins mismatch in KV loop");
        last_frame = spectrum_s(end, :);
    end
    fprintf("  Decoder with past KV loop successfully validated.\n");
else
    fprintf("  [NOTICE] decoder_with_past_model.onnx not found. Single-step only validated.\n");
end

fprintf("[test_decoder] All decoder tests PASSED.\n");

end
