function y = reconstruct_waveform(target_mel, fs)
%RECONSTRUCT_WAVEFORM Reconstruct a waveform from a linear-frequency matrix.
%   This baseline uses Griffin-Lim-style iterative phase estimation. A neural
%   HiFi-GAN vocoder can replace this function without changing the UI API.

arguments
    target_mel (:,:) {mustBeNumeric, mustBeFinite}
    fs (1,1) {mustBePositive, mustBeFinite} = 24000
end

if isempty(target_mel)
    error("reconstruct_waveform:EmptyFeatures", "Target features cannot be empty.");
end

N = 1024;
hop = 256;
window = hann(N, "periodic");
linear_spectrum = 10.^(double(target_mel) / 20);
phase = exp(1i * 2 * pi * rand(size(linear_spectrum)));
S = linear_spectrum .* phase;
for iteration = 1:32
    y = istft(S, fs, Window=window, OverlapLength=N-hop, FFTLength=N);
    [estimated, ~, ~] = stft(y, fs, Window=window, OverlapLength=N-hop, FFTLength=N);
    phase = estimated ./ max(abs(estimated), eps);
    S = linear_spectrum .* phase;
end
y = istft(S, fs, Window=window, OverlapLength=N-hop, FFTLength=N);
y = y(:);
peak = max(abs(y), [], "all");
if peak > eps
    y = y / peak;
end
end
