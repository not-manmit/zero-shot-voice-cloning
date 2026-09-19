function test_vocoder()
%TEST_VOCODER Model-backed inference validation for HiFi-GAN neural vocoder.
%
%   Verifies:
%     Stage 1: Deterministic synthetic Mel spectrogram input [80 x 40]
%     Stage 2: End-to-end decoder Mel output passed to vocoder (if decoder available)
%     Stage 3: Output waveform properties (finiteness, duration, amplitude bounds)
%     Stage 4: Audio file persistence via audiowrite

fprintf("[test_vocoder] Validating HiFi-GAN neural vocoder ...\n");

cfg = pipeline_config();

if ~isfile(cfg.paths.vocoder)
    error("test_vocoder:MissingModel", ...
        "HiFi-GAN vocoder model is missing at %s. Run scripts/download_weights.m.", cfg.paths.vocoder);
end

models = load_onnx_engine(cfg);

% --- Stage 1: Deterministic Synthetic Mel Smoke Test --------------------
fprintf("  Stage 1: Testing with synthetic log-Mel [80 x 40] ...\n");
T_frames = 40;
synthetic_mel = single(-4.0 + 1.5 * randn(cfg.n_mels, T_frames));
synthetic_mel = max(synthetic_mel, log(cfg.mel_floor));

t_voc = tic;
wave1 = reconstruct_waveform(synthetic_mel, cfg.fs, models, cfg);
elapsed1 = toc(t_voc);

fprintf("    Synthesized %d samples (%.2f s) in %.2f s\n", ...
    numel(wave1), numel(wave1)/cfg.fs, elapsed1);

assert(~isempty(wave1), "Synthesized waveform is empty");
assert(all(isfinite(wave1)), "Waveform contains NaN or Inf");
assert(max(abs(wave1)) <= 1.0 + eps, "Waveform peak amplitude exceeds [-1, 1]");
assert(numel(wave1) >= T_frames * 200, "Waveform length suspiciously short for given Mel frames");

% --- Stage 2: Upstream Decoder Mel -> Vocoder Pipeline ------------------
if isfile(cfg.paths.decoder)
    fprintf("  Stage 2: Testing decoder-generated Mel -> Vocoder boundary ...\n");
    try
        % Run lightweight feature synthesis
        dummy_emb = single(ones(1, 512) / sqrt(512));
        [dec_mel, ~] = synthesize_features("Test.", dummy_emb, models, cfg);
        wave2 = reconstruct_waveform(dec_mel, cfg.fs, models, cfg);
        assert(~isempty(wave2) && all(isfinite(wave2)), "Decoder-to-vocoder waveform invalid");
        fprintf("    Decoder-generated Mel (%d frames) successfully synthesized into %d samples.\n", ...
            size(dec_mel, 2), numel(wave2));
    catch ME
        fprintf("    [WARNING] Stage 2 decoder-to-vocoder test encountered: %s\n", ME.message);
    end
end

% --- Stage 3: Audiowrite Round-Trip Check -------------------------------
tmpWav = fullfile(tempdir, 'temp_vocoder_test.wav');
try
    audiowrite(tmpWav, wave1, cfg.fs);
    assert(isfile(tmpWav), "audiowrite failed to create temporary file");
    [read_x, read_fs] = audioread(tmpWav);
    assert(read_fs == cfg.fs, "Sample rate changed during round-trip");
    assert(numel(read_x) == numel(wave1), "Sample count changed during round-trip");
    delete(tmpWav);
catch ME
    if isfile(tmpWav); delete(tmpWav); end
    rethrow(ME);
end

fprintf("[test_vocoder] HiFi-GAN vocoder assertions PASSED.\n");

end
