function waveform = reconstruct_waveform(target_mel, fs, models, cfg)
%RECONSTRUCT_WAVEFORM Neural HiFi-GAN waveform synthesis from 80-bin Mel spectrogram.
%
%   waveform = RECONSTRUCT_WAVEFORM(target_mel, fs, models, cfg)
%
%   Boundary Contract:
%     Input:  target_mel [80 x T] natural log-Mel matrix (HTK scale, 80-7600 Hz)
%     Layout: Formatted via tensor_contract_utils as [1, 80, T] SCB for 'spectrogram'
%     Output: 16 000 Hz single-channel discrete-time speech sequence [N x 1]
%
%   Synthesis Steps:
%     1. Validate Mel dimension and frame count (must have 80 bins, >= 5 frames)
%     2. Apply log-floor bounding: max(mel, log(cfg.mel_floor))
%     3. Format dlarray input via centralized tensor_contract_utils
%     4. HiFi-GAN ONNX forward inference: predict(models.vocoder, voc_in)
%     5. Extract waveform array (256x upsampling factor: N = T * 256)
%     6. Peak normalisation to safe acoustic range [-1.0, 1.0]

arguments
    target_mel (:,:) {mustBeNumeric, mustBeFinite}
    fs (1,1) {mustBePositive, mustBeFinite} = 16000
    models struct = struct()
    cfg struct = struct()
end

if isempty(cfg) || ~isfield(cfg, 'fs')
    cfg = pipeline_config();
end
if isempty(models) || ~isfield(models, 'vocoder')
    models = load_onnx_engine(cfg);
end

if isempty(target_mel)
    error("reconstruct_waveform:EmptyFeatures", "Target Mel acoustic feature matrix is empty.");
end

% Orient to [80 x T]
mel = double(target_mel);
if size(mel, 1) ~= cfg.n_mels && size(mel, 2) == cfg.n_mels
    mel = mel.';
end

if size(mel, 1) ~= cfg.n_mels
    error("reconstruct_waveform:InvalidMelDimensions", ...
        "Mel spectrogram must have %d frequency bins (observed %d x %d).", ...
        cfg.n_mels, size(mel, 1), size(mel, 2));
end

if size(mel, 2) < 4
    error("reconstruct_waveform:MelTooShort", ...
        "Mel spectrogram has only %d temporal frames (minimum required: 4).", size(mel, 2));
end

% Ensure natural log floor bounding
mel = max(mel, log(cfg.mel_floor));

% Format dlarray using centralized utility
try
    voc_in = tensor_contract_utils.format_vocoder_inputs(mel, models.vocoder);
    out = predict(models.vocoder, voc_in);
    wave_raw = extractdata(out);
catch ME
    error("reconstruct_waveform:InferenceFailed", ...
        "HiFi-GAN vocoder inference failed.\nError: %s\nExpected input: 'spectrogram' [T_mel, 80] format UU.", ME.message);
end

waveform = double(wave_raw(:));

if isempty(waveform) || ~all(isfinite(waveform))
    error("reconstruct_waveform:InvalidWaveformOutput", ...
        "Vocoder produced an empty, NaN, or Inf waveform sequence.");
end

% Peak normalize to unity [-1, 1] without blowing up near-silent outputs
peak = max(abs(waveform));
if peak > 1.0
    waveform = waveform ./ peak;
end
waveform = max(-1.0, min(1.0, waveform));

end
