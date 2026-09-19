function spk_emb = extract_speaker_embedding(x, fs, models, cfg)
%EXTRACT_SPEAKER_EMBEDDING Derive a 512-dim L2-normalized speaker embedding via CAM++.
%
%   spk_emb = EXTRACT_SPEAKER_EMBEDDING(x, fs, models, cfg)
%
%   DSP and Deep Learning Chain:
%     1. Signal validation (non-silent, minimum duration >= 1.0 s)
%     2. Resample and precondition to 16 kHz mono
%     3. 80-bin filterbank feature extraction with per-utterance CMN
%        (25 ms periodic Hann window, 10 ms hop, 20-8000 Hz filter bank)
%     4. Utterance chunking (3.0 s window with 50% overlap)
%     5. CAM++ ONNX model inference using explicit 'features' [1, T, 80] CBT layout
%     6. Chunk embedding aggregation via mean pooling
%     7. L2 normalization: e_norm = e / ||e||_2
%
%   Outputs:
%     spk_emb – 512-dimensional L2-normalized single-precision row vector [1 x 512]

arguments
    x      (:,:) {mustBeNumeric, mustBeFinite}
    fs     (1,1) {mustBePositive, mustBeFinite}
    models struct
    cfg    struct = struct()
end

if ~isfield(cfg, 'fs')
    cfg = pipeline_config();
end

% 1. Preprocess & Validate Audio Sequence
if ~isvector(x)
    x = mean(x, 2);
end
x = x(:);

if abs(fs - cfg.fs) > 1
    [x, fs] = preprocess_signal(x, fs, cfg.fs);
end

peakVal = max(abs(x));
if peakVal > 1.0
    x = x / peakVal;
end

dur_s = numel(x) / fs;
if dur_s < cfg.spk_min_dur_s
    error("extract_speaker_embedding:AudioTooShort", ...
        "Reference audio duration is %.2f s. Minimum required is %.2f s.", dur_s, cfg.spk_min_dur_s);
end

rms_val = sqrt(mean(x.^2));
if rms_val < 1e-4
    error("extract_speaker_embedding:SilentAudio", ...
        "Reference audio appears silent (RMS = %.2e). Check microphone or recording levels.", rms_val);
end

% 2. Temporal Chunking (3.0 s with 50% overlap)
chunk_len = round(cfg.spk_chunk_dur_s * fs);
hop_len   = round(chunk_len / 2);
n_samples = numel(x);

starts = 1 : hop_len : max(1, n_samples - chunk_len + 1);
if starts(end) + chunk_len - 1 < n_samples
    starts(end+1) = n_samples - chunk_len + 1;
end
starts = unique(starts);

embeddings = zeros(numel(starts), cfg.spk_emb_dim, "single");

% 3. Extract Features and Predict Embeddings
for k = 1:numel(starts)
    s = starts(k);
    e = min(s + chunk_len - 1, n_samples);
    seg = x(s:e);
    min_len = round(cfg.spk_min_dur_s * fs);
    if numel(seg) < min_len
        seg(end+1:min_len) = 0; %#ok<AGROW>
    end

    fbank = compute_campp_fbank80(seg, fs);
    embeddings(k, :) = infer_campp(fbank, models, cfg);
end

% 4. Mean Pooling and L2 Normalization
mean_emb = mean(embeddings, 1);
l2_norm = norm(mean_emb, 2);

if l2_norm < eps
    error("extract_speaker_embedding:ZeroEmbeddingNorm", ...
        "Extracted speaker embedding has near-zero L2 norm.");
end

spk_emb = single(mean_emb ./ l2_norm);

if ~all(isfinite(spk_emb))
    error("extract_speaker_embedding:NonFiniteEmbedding", ...
        "Normalized speaker embedding contains non-finite values (NaN or Inf).");
end

end

function fbank = compute_campp_fbank80(seg, fs)
% 80-bin filterbank with 25 ms Hann window, 10 ms frame shift, and CMN
N = round(0.025 * fs);    % 400 samples
hop = round(0.010 * fs);  % 160 samples
n_fft = 512;
win = hann(N, 'periodic');

[S, ~, ~] = stft(seg, fs, Window=win, OverlapLength=N-hop, FFTLength=n_fft, FrequencyRange='onesided');
power = abs(S).^2; % [257 x T]
f_axis = linspace(0, fs/2, n_fft/2 + 1);

mel_power = apply_triangular_filters(power, f_axis, 80, 20, 8000);
log_mel = log(max(mel_power, 1e-10));

% Cepstral Mean Normalization (CMN) per segment (zero-mean over temporal frames)
cmn_mel = log_mel - mean(log_mel, 2);
fbank = single(cmn_mel'); % [T x 80]
end

function mel_out = apply_triangular_filters(power, f, n_mels, f_min, f_max)
mel_min = 2595 * log10(1 + f_min / 700);
mel_max = 2595 * log10(1 + f_max / 700);
mel_pts = linspace(mel_min, mel_max, n_mels + 2);
hz_pts  = 700 * (10.^(mel_pts / 2595) - 1);

n_bins = numel(f);
bin_pts = zeros(1, n_mels + 2);
for k = 1:n_mels + 2
    [~, bin_pts(k)] = min(abs(f - hz_pts(k)));
end

mel_out = zeros(n_mels, size(power, 2));
for m = 1:n_mels
    bL = max(1, bin_pts(m));
    bC = max(bL + 1, min(bin_pts(m+1), n_bins - 1));
    bR = min(n_bins, bin_pts(m+2));
    bC = max(bL + 1, min(bR - 1, bC));

    nRise = bC - bL + 1;
    if nRise >= 1
        rise = linspace(0, 1, nRise)';
        mel_out(m, :) = mel_out(m, :) + rise' * power(bL:bC, :);
    end

    nFall = bR - bC;
    if nFall >= 1
        fall = linspace(1, 0, nFall + 1)';
        fall = fall(2:end);
        mel_out(m, :) = mel_out(m, :) + fall' * power(bC+1:bR, :);
    end
end
end

function emb = infer_campp(fbank, models, cfg)
if ~isfield(models, 'spk_encoder') || isempty(models.spk_encoder)
    error("extract_speaker_embedding:MissingModel", ...
        "CAM++ speaker encoder is not loaded in models struct. Run setup_matlab_online.m.");
end

net = models.spk_encoder;

% Format tensor via centralized utility
try
    inp = tensor_contract_utils.format_campp_inputs(fbank, net);
    out = predict(net, inp);
    emb_raw = extractdata(out);
catch ME
    error("extract_speaker_embedding:InferenceFailure", ...
        "CAM++ speaker encoder inference failed.\nError: %s\nExpected input: 'features' [1, T, 80] CBT. Check models/xvector/xvector_encoder.onnx.", ...
        ME.message);
end

emb = single(emb_raw(:)');
if numel(emb) ~= cfg.spk_emb_dim
    error("extract_speaker_embedding:DimensionMismatch", ...
        "CAM++ output dimension %d does not match expected cfg.spk_emb_dim=%d.", numel(emb), cfg.spk_emb_dim);
end
end
