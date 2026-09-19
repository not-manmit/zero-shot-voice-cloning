function test_end_to_end()
%TEST_END_TO_END Full pipeline: ref -> embedding -> tokens -> TTS -> vocoder -> waveform

cfg = pipeline_config();
required = {cfg.paths.encoder, cfg.paths.decoder, cfg.paths.decoder_kv, cfg.paths.vocoder, cfg.paths.spk_encoder};
for i=1:numel(required)
    assert(isfile(required{i}), sprintf('Required model assets are unavailable: %s', required{i}));
end

models = load_onnx_engine(cfg);
fs = cfg.fs;
% 2 s synthetic reference
t = (0:1/fs:2-1/fs)';
refAudio = 0.6*sin(2*pi*180*t) + 0.2*sin(2*pi*240*t) + 0.05*randn(size(t));
refAudio = refAudio / max(abs(refAudio));

result = generate_voice(refAudio, fs, 'Hello, this is a zero-shot voice cloning demonstration in MATLAB.', models, cfg);
assert(~isempty(result.waveform), 'Waveform empty');
assert(all(isfinite(result.waveform)), 'Waveform non-finite');
assert(result.sampleRate == cfg.fs, 'Sample rate mismatch');
assert(numel(result.waveform) > 0 && numel(result.waveform)/result.sampleRate > 0.3, 'Waveform duration too short');
assert(max(abs(result.waveform)) <= 1+1e-6, 'Waveform amplitude invalid');
assert(~isempty(result.speakerEmbedding) && numel(result.speakerEmbedding)==cfg.spk_emb_dim, 'Bad embedding');
assert(~isempty(result.acousticFeatures) && size(result.acousticFeatures,1)==cfg.n_mels, 'Bad acoustic');
assert(all(isfinite(result.acousticFeatures),'all'), 'Acoustic non-finite');
% Verify it can be written
tmp = fullfile(tempdir,'test_e2e.wav');
audiowrite(tmp, result.waveform, result.sampleRate);
assert(isfile(tmp), 'audiowrite e2e failed');
delete(tmp);

fprintf('[test_end_to_end] OK: %d samples @%dHz, %d mel frames, total %.2fs\n', ...
    numel(result.waveform), result.sampleRate, size(result.acousticFeatures,2), result.metrics.totalTime_s);
end
