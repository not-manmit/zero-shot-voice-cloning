function [mel_matrix, S, f, t] = stft_analysis(x, fs, num_mel_bands)
%STFT_ANALYSIS Compute a Hann-windowed STFT and Mel spectro-temporal matrix.
%   mel_matrix = STFT_ANALYSIS(x, fs) returns a dB-scaled Mel matrix.
%   [mel_matrix, S, f, t] also returns the complex STFT and its axes.

arguments
    x (:,1) {mustBeNumeric, mustBeFinite}
    fs (1,1) {mustBePositive, mustBeFinite}
    num_mel_bands (1,1) {mustBeInteger, mustBePositive} = 80
end

N = 1024;
hop = 256;
window = hann(N, "periodic");
[S, f, t] = stft(double(x), fs, Window=window, OverlapLength=N-hop, FFTLength=N);

power_spectrum = abs(S).^2;
mel_matrix = mel_filter_bank(power_spectrum, fs, N, num_mel_bands);
mel_matrix = 10 * log10(max(mel_matrix, eps));
end

function mel_matrix = mel_filter_bank(power_spectrum, fs, N, num_mel_bands)
%MEL_FILTER_BANK Apply triangular filters uniformly spaced in Mel scale.
low_mel = 2595 * log10(1 + 0 / 700);
high_mel = 2595 * log10(1 + (fs / 2) / 700);
mel_points = linspace(low_mel, high_mel, num_mel_bands + 2);
frequencies = 700 * (10.^(mel_points / 2595) - 1);
bins = floor((N + 1) * frequencies / fs) + 1;
mel_matrix = zeros(num_mel_bands, size(power_spectrum, 2));

for band = 1:num_mel_bands
    left = max(1, bins(band));
    center = max(left + 1, bins(band + 1));
    right = min(size(power_spectrum, 1), max(center + 1, bins(band + 2)));
    rising = linspace(0, 1, center - left + 1);
    falling = linspace(1, 0, right - center + 1);
    mel_matrix(band, :) = rising * power_spectrum(left:center, :) + ...
        falling(2:end) * power_spectrum(center+1:right, :);
end
end
