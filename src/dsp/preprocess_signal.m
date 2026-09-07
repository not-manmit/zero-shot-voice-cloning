function [x_clean, fs] = preprocess_signal(x, fs, target_fs)
%PREPROCESS_SIGNAL Prepare a discrete-time speech sequence for the pipeline.
%   [x_clean, fs] = PREPROCESS_SIGNAL(x, fs) converts x to mono, resamples
%   it to 24 kHz, suppresses stationary background noise, and peak-normalizes
%   the result to [-1, 1].
%
%   The input x may be a row/column vector or a multi-channel matrix. The
%   first 0.5 seconds are treated as a noise-only estimate when available.

arguments
    x (:,:) {mustBeNumeric, mustBeFinite}
    fs (1,1) {mustBePositive, mustBeFinite}
    target_fs (1,1) {mustBePositive, mustBeFinite} = 24000
end

if isempty(x)
    error("preprocess_signal:EmptyInput", "Input audio cannot be empty.");
end

x = double(x);
if ~isvector(x)
    x = mean(x, 2);
else
    x = x(:);
end

% Resampling applies an anti-aliasing low-pass filter before decimation.
if fs ~= target_fs
    x = resample(x, target_fs, fs);
end
fs = target_fs;

x = x - mean(x);
x = suppress_stationary_noise(x, fs);

peak = max(abs(x), [], "all");
if peak > eps
    x_clean = x ./ peak;
else
    x_clean = zeros(size(x));
end
x_clean = max(-1, min(1, x_clean));
end

function y = suppress_stationary_noise(x, fs)
%SUPPRESS_STATIONARY_NOISE Apply a conservative STFT spectral gate.
N = 1024;
hop = 256;
window = hann(N, "periodic");
noise_length = min(numel(x), max(1, round(0.5 * fs)));
noise = x(1:noise_length);

[noise_spectrum, ~, ~] = stft(noise, fs, Window=window, OverlapLength=N-hop, FFTLength=N);
noise_floor = median(abs(noise_spectrum), 2);

[S, ~, ~] = stft(x, fs, Window=window, OverlapLength=N-hop, FFTLength=N);
magnitude = abs(S);
phase = angle(S);
threshold = max(noise_floor * 1.5, eps);
gain = max(0.08, min(1, magnitude ./ threshold));

S_filtered = gain .* magnitude .* exp(1i * phase);
y = istft(S_filtered, fs, Window=window, OverlapLength=N-hop, FFTLength=N);
y = y(:);
y = y(1:min(numel(y), numel(x)));
if numel(y) < numel(x)
    y(end+1:numel(x), 1) = 0;
end
end
