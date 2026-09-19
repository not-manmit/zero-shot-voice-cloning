function spk_emb = extract_speaker_embedding(x, fs, models, cfg)
%EXTRACT_SPEAKER_EMBEDDING  Derive a 512-dim L2-normalised CAM++ embedding.
%
%   spk_emb = EXTRACT_SPEAKER_EMBEDDING(x, fs, models, cfg)
%
%   Uses openspeech/wespeaker CAM++ ONNX (voxceleb_CAM++.onnx) which expects
%   80-dim log-Mel filterbank with per-utterance CMN (25 ms win, 10 ms hop).
%   Long clips are chunked 3 s with 50% overlap, embeddings averaged and
%   L2-normalised.  Explicit input name contract: 'features' [B,T,80].

arguments
    x      (:,:) {mustBeNumeric, mustBeFinite}
    fs     (1,1) {mustBePositive, mustBeFinite}
    models struct
    cfg    struct = struct()
end

if ~isfield(cfg, 'fs')
    cfg = pipeline_config();
end

% --- 1. Mono + 16 kHz ---------------------------------------------------
if ~isvector(x)
    x = mean(x, 2);
end
x = x(:);
if abs(fs - cfg.fs) > 1
    [x, fs] = preprocess_signal(x, fs, cfg.fs);
end
if max(abs(x)) > 1
    x = x / max(abs(x));
end

% --- 2. Duration / silence ---------------------------------------------
dur_s = numel(x) / fs;
if dur_s < cfg.spk_min_dur_s
    error("extract_speaker_embedding:TooShort", ...
        "Reference audio is %.2f s; minimum required is %.2f s.", dur_s, cfg.spk_min_dur_s);
end
rms_val = sqrt(mean(x .^ 2));
if rms_val < 1e-4
    error("extract_speaker_embedding:SilentAudio", ...
        "Reference audio appears silent (RMS = %.2e).", rms_val);
end
if ~all(isfinite(x))
    error("extract_speaker_embedding:NonFiniteInput", "Reference contains NaN/Inf.");
end

% --- 3. Chunk -----------------------------------------------------------
chunk_len = round(cfg.spk_chunk_dur_s * fs);
hop_len   = round(chunk_len / 2);
n_samples = numel(x);
starts = 1 : hop_len : max(1, n_samples - chunk_len + 1);
if starts(end) + chunk_len -1 < n_samples
    starts(end+1) = n_samples - chunk_len + 1;
end
starts = unique(starts);

embeddings = zeros(numel(starts), cfg.spk_emb_dim, "single");

for k = 1:numel(starts)
    s = starts(k);
    e = min(s + chunk_len - 1, n_samples);
    seg = x(s:e);
    min_len = round(cfg.spk_min_dur_s * fs);
    if numel(seg) < min_len
        seg(end+1:min_len) = 0;
    end
    fbank = compute_fbank80(seg, fs, cfg); % [T,80]
    embeddings(k, :) = run_campp_onnx(fbank, models, cfg);
end

% --- 4. Average + L2 norm -----------------------------------------------
mean_emb = mean(embeddings, 1); % [1,512]
nrm = norm(mean_emb, 2);
if nrm < eps
    error("extract_speaker_embedding:ZeroEmbedding", "Embedding has near-zero norm after averaging.");
end
spk_emb = single(mean_emb ./ nrm);
if ~all(isfinite(spk_emb))
    error("extract_speaker_embedding:NonFiniteEmbedding", "Embedding contains NaN/Inf after norm.");
end
fprintf('[extract_speaker_embedding] Computed %d chunks -> embedding L2=%.3f dim %d\n', numel(starts), norm(spk_emb), numel(spk_emb));
end

% -----------------------------------------------------------------------
function fbank = compute_fbank80(x_seg, fs, cfg)
% fbank80 with 25ms win 10ms hop, 20-8000 Hz, log, CMN – matches CAM++
N = round(0.025*fs); % 400 at 16kHz
hop = round(0.010*fs); % 160
n_fft = 512;
win = hann(N,'periodic'); % Hann per DSP spec (consistent with stft_analysis)
% STFT one-sided, same as Wespeaker Kaldi fbank pipeline
[S,~,~] = stft(x_seg, fs, Window=win, OverlapLength=N-hop, FFTLength=n_fft, FrequencyRange='onesided');
power = abs(S).^2; % [257 x T]
f = linspace(0, fs/2, n_fft/2+1);
mel_out = apply_mel(power, f, 80, 20, 8000);
mel_out = log(max(mel_out, 1e-10));
% CMN per utterance (zero-mean over time) — required by CAM++
mel_out = mel_out - mean(mel_out,2);
fbank = single(mel_out'); % [T,80]
end

function mel_out = apply_mel(power, f, n_mels, f_min, f_max)
mel_min = 2595*log10(1+f_min/700);
mel_max = 2595*log10(1+f_max/700);
mel_pts = linspace(mel_min, mel_max, n_mels+2);
hz_pts = 700*(10.^(mel_pts/2595)-1);
bin_pts = zeros(1,n_mels+2);
for k=1:n_mels+2
    [~,bin_pts(k)] = min(abs(f - hz_pts(k)));
end
n_bins = numel(f);
mel_out = zeros(n_mels, size(power,2));
for m=1:n_mels
    bL = max(1, bin_pts(m));
    bC = max(bL+1, min(bin_pts(m+1), n_bins-1));
    bR = min(n_bins, bin_pts(m+2));
    bC = max(bL+1, min(bR-1, bC));
    nRise = bC-bL+1;
    if nRise>=1
        rise = linspace(0,1,nRise)';
        mel_out(m,:) = mel_out(m,:) + rise'*power(bL:bC,:);
    end
    nFall = bR-bC;
    if nFall>=1
        fall = linspace(1,0,nFall+1)';
        fall = fall(2:end);
        mel_out(m,:) = mel_out(m,:) + fall'*power(bC+1:bR,:);
    end
end
end

function emb = run_campp_onnx(fbank, models, cfg)
% fbank [T,80] -> embedding [1,512]  Explicit contract: input name 'features'
x_f32 = single(fbank);
if ~isfield(models,'spk_encoder') || isempty(models.spk_encoder)
    error("extract_speaker_embedding:MissingModel","No speaker encoder loaded. Run setup_matlab_online.m");
end
net = models.spk_encoder;
names = cellstr(net.InputNames);
% Explicit contract check — no heuristic fallback
if ~any(strcmpi(names,'features'))
    error("extract_speaker_embedding:ContractMismatch", ...
        "Speaker encoder ONNX expected input 'features' but found [%s]. Check voxceleb_CAM++.onnx export.", strjoin(names,','));
end
% Verified: Wespeaker CAM++ expects [B,T,80] float32, input name 'features', single verified layout CBT [1,T,80].
% Fallback removed — production uses one contract; diagnose_onnx must confirm.
try
    inp = dlarray(reshape(x_f32, 1, size(x_f32,1), size(x_f32,2)), 'CBT');
    out = predict(net, inp);
    emb_raw = extractdata(out);
catch ME
    error("extract_speaker_embedding:InferenceFailed", ...
        "CAM++ ONNX inference failed with verified layout [1,T,80] CBT.\nError: %s\nCheck that %s is voxceleb_CAM++.onnx (512-dim) and Run diagnose_onnx to confirm input name ''features''.", ...
        ME.message, cfg.paths.spk_encoder);
end
emb = single(emb_raw(:)');
if numel(emb) ~= cfg.spk_emb_dim
    error("extract_speaker_embedding:DimMismatch","CAM++ output %d dim, expected %d. Check ONNX is voxceleb_CAM++.onnx", numel(emb), cfg.spk_emb_dim);
end
% Validate L2 will be applied by caller; raw embedding should be finite
if ~all(isfinite(emb))
    error("extract_speaker_embedding:NonFiniteRaw","CAM++ raw embedding contains NaN/Inf");
end
end
