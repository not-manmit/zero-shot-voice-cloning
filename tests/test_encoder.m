function test_encoder()
%TEST_ENCODER Model-backed smoke test for the SpeechT5 text encoder ONNX model.
%
%   Verifies:
%     1. Model file existence at canonical path
%     2. Import via centralized import_onnx_model
%     3. Inference with real token sequence and attention mask
%     4. Output dimension consistency: last_hidden_state [1, T, 768]
%     5. Output finiteness (no NaN or Inf)

fprintf("[test_encoder] Validating SpeechT5 text encoder ...\n");

cfg = pipeline_config();

if ~isfile(cfg.paths.encoder)
    error("test_encoder:ModelMissing", ...
        "SpeechT5 encoder ONNX model is not present:\n  %s\nRun scripts/setup_matlab_online.m to download.", cfg.paths.encoder);
end

% 1. Import Encoder
[encoderNet, meta] = import_onnx_model(cfg.paths.encoder);
assert(~isempty(encoderNet), "Encoder network object is empty");

% 2. Prepare Valid Token Sequence
test_text = "The quick brown fox jumps over the lazy dog.";
[token_ids, attn_mask] = tokenize_text(test_text, cfg);
T = numel(token_ids);
fprintf("  Input sequence length: %d tokens (with EOS)\n", T);

% 3. Format dlarray Inputs
enc_inputs = tensor_contract_utils.format_encoder_inputs(token_ids, attn_mask, encoderNet);

% 4. Run Forward Inference
t_infer = tic;
try
    enc_raw_out = predict(encoderNet, enc_inputs);
    elapsed = toc(t_infer);
catch ME
    error("test_encoder:InferenceFailed", ...
        "Encoder forward pass failed: %s\nInputs: [%s]", ME.message, strjoin(meta.inputNames, ", "));
end

% 5. Inspect and Validate Output Dimensions
hidden_states = tensor_contract_utils.parse_encoder_outputs(enc_raw_out, encoderNet);
sz = size(hidden_states);

fprintf("  Inference succeeded in %.2f s. Hidden states shape: %s\n", elapsed, mat2str(sz));

assert(all(isfinite(hidden_states), 'all'), "Hidden state output contains NaN or Inf");
assert(sz(1) == 1, "Batch dimension must equal 1");
assert(sz(2) == T, sprintf("Temporal sequence dimension (%d) must match token count (%d)", sz(2), T));
assert(sz(3) == cfg.encoder_hidden, sprintf("Hidden dimension (%d) must match cfg.encoder_hidden (%d)", sz(3), cfg.encoder_hidden));

fprintf("[test_encoder] Encoder smoke test PASSED.\n");

end
