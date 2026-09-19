function waveform = reconstruct_waveform(target_mel, fs, models, cfg)
%RECONSTRUCT_WAVEFORM Reconstruct waveform using the model-specific neural vocoder.
%
% The project does not substitute a heuristic phase-reconstruction algorithm for
% the selected SpeechT5 neural vocoder. When the ONNX vocoder is available, the
% function calls the model directly and validates the output before returning it.

arguments
    target_mel (:,:) {mustBeNumeric, mustBeFinite}
    fs (1,1) {mustBePositive, mustBeFinite} = 16000
    models struct = struct()
    cfg struct = struct()
end

if isempty(cfg)
    cfg = pipeline_config();
end
if isempty(models) || ~isfield(models, 'vocoder')
    models = load_onnx_engine(cfg);
end

if isempty(target_mel)
    error('reconstruct_waveform:EmptyFeatures', 'Target features cannot be empty.');
end
if ~isfield(models, 'vocoder') || isempty(models.vocoder)
    error('reconstruct_waveform:MissingModel', 'No vocoder model is available. Run setup_matlab_online.m and ensure models/speecht5/vocoder_model.onnx exists.');
end

mel_in = double(target_mel);
if size(mel_in, 1) ~= cfg.n_mels && size(mel_in, 2) == cfg.n_mels
    mel_in = mel_in.';
end

try
    vocoder_in = dlarray(single(mel_in), 'CB');
    out = predict(models.vocoder, vocoder_in);
    waveform = double(extractdata(out));
catch ME
    % If the model requires a different shape or ordering, fall back to an
    % explicit batch-initial channel layout before failing. This preserves the
    % runtime contract check without inventing a fake waveform synthesis.
    try
        vocoder_in = dlarray(single(reshape(mel_in, 1, size(mel_in, 1), size(mel_in, 2))), 'SCB');
        out = predict(models.vocoder, vocoder_in);
        waveform = double(extractdata(out));
    catch
        error('reconstruct_waveform:InferenceFailed', 'Neural vocoder inference failed: %s', ME.message);
    end
end

waveform = waveform(:);
waveform = waveform(~isnan(waveform) & ~isinf(waveform));
if isempty(waveform)
    error('reconstruct_waveform:InvalidOutput', 'Vocoder output was empty or non-finite.');
end

peak = max(abs(waveform), [], 'all');
if peak > 0
    waveform = waveform ./ peak;
end
waveform = double(waveform);
end
