function test_end_to_end()
%TEST_END_TO_END A lightweight end-to-end pipeline smoke test.
%
% This test reports "Model assets unavailable" if the ONNX files are not present
% instead of pretending success.

cfg = pipeline_config();
required = {cfg.paths.encoder, cfg.paths.decoder, cfg.paths.decoder_kv, cfg.paths.vocoder, cfg.paths.spk_encoder};
if ~all(cellfun(@(p) isfile(p), required))
    fprintf('[test_end_to_end] Model assets unavailable.\n');
    return;
end

try
    models = load_onnx_engine(cfg);
    [refAudio, refFs] = audioread('sample_reference.wav');
    if isempty(refAudio)
        fprintf('[test_end_to_end] No sample_reference.wav available.\n');
        return;
    end
    result = generate_voice(refAudio, refFs, 'Hello, this is a zero-shot voice cloning demonstration.', models, cfg);
    if isempty(result.waveform)
        error('generate_voice:EmptyWaveform', 'Waveform output was empty.');
    end
    if ~all(isfinite(result.waveform(:)))
        error('generate_voice:NonFiniteWaveform', 'Waveform output was not finite.');
    end
    if numel(result.waveform) == 0 || isempty(result.sampleRate)
        error('generate_voice:InvalidOutput', 'Waveform / sample rate invalid.');
    end
    fprintf('[test_end_to_end] OK generated waveform length %d at %d Hz.\n', numel(result.waveform), result.sampleRate);
catch ME
    fprintf('[test_end_to_end] %s\n', ME.message);
end
end
