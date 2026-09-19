function waveform = reconstruct_waveform(target_mel, fs, models, cfg)
%RECONSTRUCT_WAVEFORM  Neural HiFi-GAN vocoder with explicit contract.
%
%   waveform = RECONSTRUCT_WAVEFORM(target_mel, fs, models, cfg)
%
%   Contract: input Mel [80,T] or [T,80] will be oriented to [1,80,T] float32
%   with input name 'spectrogram'. Output: mono 16 kHz waveform [N,1].
%   Xenova/speecht5_hifigan model.onnx expects natural-log Mel (not dB).

arguments
    target_mel (:,:) {mustBeNumeric, mustBeFinite}
    fs (1,1) {mustBePositive, mustBeFinite} = 16000
    models struct = struct()
    cfg struct = struct()
end

if isempty(cfg) || ~isfield(cfg,'fs')
    cfg = pipeline_config();
end
if isempty(models) || ~isfield(models, 'vocoder')
    models = load_onnx_engine(cfg);
end
if isempty(target_mel)
    error('reconstruct_waveform:EmptyFeatures', 'Mel input is empty.');
end
if ~isfield(models,'vocoder') || isempty(models.vocoder)
    error('reconstruct_waveform:MissingModel', 'No vocoder loaded – run setup_matlab_online.m');
end

% Orient to [80,T]
mel = double(target_mel);
if size(mel,1) ~= cfg.n_mels && size(mel,2) == cfg.n_mels
    mel = mel.';
end
if size(mel,1) ~= cfg.n_mels
    error('reconstruct_waveform:BadMelShape','Mel must have %d bins, got %d x %d (expected [80,T])', cfg.n_mels, size(mel,1), size(mel,2));
end
if size(mel,2) < 5
    error('reconstruct_waveform:MelTooShort','Mel has only %d frames; need >=5', size(mel,2));
end
if abs(fs - cfg.hifigan_sr) > 1
    warning('reconstruct_waveform:SampleRateMismatch','Vocoder sample rate %d != hifigan_sr %d — output will be at %d Hz', fs, cfg.hifigan_sr, cfg.hifigan_sr);
    fs = cfg.hifigan_sr;
end

% Clip Mel to log floor to avoid vocoder blowup/NaN
mel = max(mel, log(cfg.mel_floor));

% Explicit vocoder input contract — no heuristic fallback
names = cellstr(models.vocoder.InputNames);
if ~any(strcmpi(names,'spectrogram'))
    error('reconstruct_waveform:ContractMismatch','Vocoder expected input ''spectrogram'' but found [%s]. Check models/speecht5/vocoder_model.onnx is Xenova/speecht5_hifigan model.onnx', strjoin(names,','));
end
mel_batched = single(reshape(mel, 1, size(mel,1), size(mel,2))); % [1,80,T]
voc_in = dlarray(mel_batched, 'SCB'); % verified single layout [B,80,T] SCB; diagnose_onnx must confirm


try
    out = predict(models.vocoder, voc_in);
    wave_raw = extractdata(out);
catch ME
    error('reconstruct_waveform:InferenceFailed', 'HiFi-GAN vocoder failed: %s\nCheck that models/speecht5/vocoder_model.onnx is Xenova/speecht5_hifigan model.onnx (input spectrogram [1,80,T] log-Mel).', ME.message);
end

waveform = double(wave_raw(:));
if isempty(waveform) || ~all(isfinite(waveform))
    error('reconstruct_waveform:InvalidOutput','Vocoder output empty or non-finite');
end
if numel(waveform) < fs*0.1
    warning('reconstruct_waveform:ShortOutput','Vocoder waveform only %d samples (%.2f s)', numel(waveform), numel(waveform)/fs);
end
% Peak normalise to [-1,1] without amplifying quiet outputs
pk = max(abs(waveform));
if pk > eps
    waveform = waveform / max(pk, 1);
end
waveform = max(-1,min(1,waveform));
end
