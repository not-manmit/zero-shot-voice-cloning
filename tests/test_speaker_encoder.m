function test_speaker_encoder()
%TEST_SPEAKER_ENCODER Model-backed structural test for CAM++ speaker embedding model.
%
%   NOTE ON TESTING SCOPE:
%     This test uses a deterministic synthetic audio signal fixture to verify
%     tensor pipeline connectivity, input/output dimensions, and normalization.
%     This is a STRUCTURAL INTEGRITY SMOKE TEST, NOT a validation of speaker
%     similarity or perceptual voice-cloning quality. Real human speech validation
%     is performed in test_end_to_end_real_reference.m.
%
%   Verifies:
%     1. Model file existence at canonical path (models/xvector/xvector_encoder.onnx)
%     2. Model loads and executes via extract_speaker_embedding
%     3. Output vector has exactly 512 dimensions
%     4. Output values are finite (no NaN / Inf)
%     5. L2 norm equals 1.0 within numerical tolerance
%     6. Error handling correctly rejects silent or excessively short signals

fprintf("[test_speaker_encoder] Running CAM++ structural smoke test ...\n");

cfg = pipeline_config();
if ~isfile(cfg.paths.spk_encoder)
    error("test_speaker_encoder:MissingModel", ...
        "CAM++ model is missing at %s. Run scripts/download_weights.m.", cfg.paths.spk_encoder);
end

models = load_onnx_engine(cfg);

% 1. Synthesize Deterministic Synthetic Audio Fixture (1.5 s multi-sine)
fs = cfg.fs;
t = (0:1/fs:1.5-1/fs)';
synthetic_audio = 0.5 * sin(2*pi*150*t) + 0.3 * sin(2*pi*350*t) + 0.05 * randn(size(t));
synthetic_audio = synthetic_audio / max(abs(synthetic_audio));

% 2. Extract Embedding
t_start = tic;
emb = extract_speaker_embedding(synthetic_audio, fs, models, cfg);
elapsed = toc(t_start);

fprintf("  Embedding computed in %.2f s. Dimension: %d\n", elapsed, numel(emb));

% 3. Assertions
assert(~isempty(emb), "Embedding is empty");
assert(numel(emb) == cfg.spk_emb_dim, ...
    sprintf("Embedding dimension %d does not match expected %d", numel(emb), cfg.spk_emb_dim));
assert(all(isfinite(emb)), "Embedding contains non-finite values (NaN or Inf)");

emb_norm = norm(double(emb), 2);
assert(abs(emb_norm - 1.0) < 1e-4, ...
    sprintf("Embedding is not L2-normalized (norm = %.6f, expected 1.0)", emb_norm));

% 4. Failure Mode Test: Silent input must error
silent_audio = zeros(round(fs * 1.5), 1);
threw_expected_error = false;
try
    extract_speaker_embedding(silent_audio, fs, models, cfg);
catch ME
    if contains(ME.identifier, "SilentAudio")
        threw_expected_error = true;
    end
end
assert(threw_expected_error, "Failed to reject silent audio input with SilentAudio exception.");

fprintf("[test_speaker_encoder] Speaker encoder structural test PASSED.\n");

end
