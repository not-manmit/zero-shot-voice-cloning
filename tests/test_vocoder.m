function test_vocoder()
%TEST_VOCODER Validate HiFi-GAN vocoder with assertions.

cfg = pipeline_config();
if ~isfile(cfg.paths.vocoder)
    error('test_vocoder:MissingModel', 'Required model assets are unavailable: %s', cfg.paths.vocoder);
end

models = load_onnx_engine(cfg);
mel = randn(cfg.n_mels, 40, 'single')*2; % [80,40]
mel = max(mel, log(cfg.mel_floor));
waveform = reconstruct_waveform(mel, cfg.fs, models, cfg);
assert(~isempty(waveform), 'Waveform empty');
assert(all(isfinite(waveform)), 'Waveform non-finite');
assert(numel(waveform) > cfg.fs*0.2, 'Waveform suspiciously short');
assert(max(abs(waveform)) <= 1+1e-6, 'Waveform not normalised');
% audiowrite round-trip
tmp = fullfile(tempdir,'test_vocoder_output.wav');
audiowrite(tmp, waveform, cfg.fs);
assert(isfile(tmp), 'audiowrite failed');
delete(tmp);

fprintf('[test_vocoder] OK: %d samples @ %d Hz\n', numel(waveform), cfg.fs);
end
