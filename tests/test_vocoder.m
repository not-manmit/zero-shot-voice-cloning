function test_vocoder()
%TEST_VOCODER Validate the neural vocoder interface and waveform output contract.

cfg = pipeline_config();
if ~isfile(cfg.paths.vocoder)
    fprintf('[test_vocoder] Model assets unavailable.\n');
    return;
end

try
    models = load_onnx_engine(cfg);
    mel = rand(cfg.n_mels, 20, 'single');
    waveform = reconstruct_waveform(mel, cfg.fs, models, cfg);
    if isempty(waveform)
        error('reconstruct_waveform:EmptyWaveform', 'Waveform output was empty.');
    end
    if ~all(isfinite(waveform(:)))
        error('reconstruct_waveform:NonFiniteWaveform', 'Waveform contained NaN or Inf values.');
    end
    fprintf('[test_vocoder] OK waveform length %d, sample rate %d.\n', numel(waveform), cfg.fs);
catch ME
    fprintf('[test_vocoder] %s\n', ME.message);
end
end
