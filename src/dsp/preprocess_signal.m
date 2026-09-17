function [x_clean, fs] = preprocess_signal(x, fs, target_fs)
%PREPROCESS_SIGNAL Prepare a discrete-time speech sequence for the pipeline.
%   [x_clean, fs] = PREPROCESS_SIGNAL(x, fs) converts x to mono, resamples
%   to 16 kHz (SpeechT5 native rate), suppresses stationary background
%   noise via spectral gating, and peak-normalises the result to [-1, 1].
%
%   [x_clean, fs] = PREPROCESS_SIGNAL(x, fs, target_fs) overrides the
%   default 16 000 Hz target – useful for stand-alone DSP demonstrations.
%
%   The input x may be a row/column vector or a multi-channel matrix.
%   The first 0.5 s is used as a noise-only reference when available.
%
%   Signal-processing chain
%   -----------------------
%   1.  Mono mix-down  (mean across channels)
%   2.  Nyquist–Shannon anti-alias resample to target_fs
%   3.  DC bias removal  (subtract mean)
%   4.  Stationary-noise spectral gate  (STFT domain)
%   5.  Peak normalisation to unity amplitude
%   6.  Hard clip to [-1, 1]  (guard against float rounding)
%
%   Validation
%   ----------
%   - Empty input  → error
%   - All-zero / silent signal  → zero output (no error)
%   - NaN / Inf samples  → error (caught by mustBeFinite)
%   - Short clips (< 1 s)  → noise gate skipped gracefully

arguments
    x (:,:) {mustBeNumeric, mustBeFinite}
    fs (1,1) {mustBePositive, mustBeFinite}
    target_fs (1,1) {mustBePositive, mustBeFinite} = 16000
end

if isempty(x)
    error("preprocess_signal:EmptyInput", "Input audio cannot be empty.");
end

% --- 1. Mono mix-down -----------------------------------------------
x = double(x);
if ~isvector(x)
    x = mean(x, 2);          % average across channels (columns)
else
    x = x(:);                 % ensure column vector
end

% --- 2. Resample to target_fs (Nyquist–Shannon anti-alias filter) ----
if fs ~= target_fs
    [p, q] = rat(target_fs / fs, 1e-4);   % rational approximation P/Q
    x = resample(x, p, q);                 % polyphase anti-alias filter
end
fs = target_fs;

% --- 3. DC bias removal ---------------------------------------------
x = x - mean(x);

% --- 4. Spectral noise gate -----------------------------------------
x = suppress_stationary_noise(x, fs);

% --- 5. Peak normalisation ------------------------------------------
peak = max(abs(x), [], "all");
if peak > eps
    x_clean = x ./ peak;
else
    % Silent or near-silent audio – return zero vector without error.
    x_clean = zeros(size(x));
    return
end

% --- 6. Hard clip ---------------------------------------------------
x_clean = max(-1, min(1, x_clean));
end

function y = suppress_stationary_noise(x, fs)
%SUPPRESS_STATIONARY_NOISE Conservative STFT-domain spectral gate.
%   Uses the first 0.5 s as a noise reference.  Gain is clamped to
%   [0.08, 1] so that speech energy is never completely zeroed out.
%   For clips shorter than 0.5 s the noise floor is estimated from the
%   entire signal, which degrades suppression quality gracefully.

N   = 1024;
hop = 256;
window = hann(N, "periodic");

noise_samples = min(numel(x), max(N, round(0.5 * fs)));
noise_seg     = x(1:noise_samples);

[Sn, ~, ~]   = stft(noise_seg, fs, Window=window, ...
                     OverlapLength=N-hop, FFTLength=N);
noise_floor  = median(abs(Sn), 2);            % per-frequency median power

[S, ~, ~]    = stft(x, fs, Window=window, ...
                    OverlapLength=N-hop, FFTLength=N);
magnitude    = abs(S);
phase        = angle(S);
threshold    = max(noise_floor * 1.5, eps);    % per-bin threshold
gain         = max(0.08, min(1, magnitude ./ threshold));

S_filtered   = gain .* magnitude .* exp(1i .* phase);
y            = istft(S_filtered, fs, Window=window, ...
                     OverlapLength=N-hop, FFTLength=N);
y            = y(:);

% Trim / zero-pad to original length
n_in = numel(x);
if numel(y) > n_in
    y = y(1:n_in);
elseif numel(y) < n_in
    y(end+1:n_in, 1) = 0;
end
end
