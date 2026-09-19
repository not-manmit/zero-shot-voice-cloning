function [x_clean, fs, dsp_info] = preprocess_signal(x, fs, target_fs, options)
%PREPROCESS_SIGNAL Discrete-time speech preconditioning and conditioning chain.
%
%   [x_clean, fs] = PREPROCESS_SIGNAL(x, fs)
%   [x_clean, fs, dsp_info] = PREPROCESS_SIGNAL(x, fs, target_fs)
%   [x_clean, fs, dsp_info] = PREPROCESS_SIGNAL(x, fs, target_fs, options)
%
%   Signals & Systems Foundations:
%   ------------------------------
%   1. Multi-channel spatial down-mixing (spatial averaging to mono)
%   2. Nyquist–Shannon anti-aliasing polyphase rational resampling to target_fs
%   3. Zero-frequency (DC) bias removal: y[n] = x[n] - E[x[n]]
%   4. Short-Time Fourier Transform (STFT) spectral analysis with periodic Hann windowing
%   5. Safe, conservative spectral-domain noise attenuation (energy-gated)
%   6. Inverse Short-Time Fourier Transform (ISTFT) synthesis via overlap-add
%   7. Peak amplitude normalization to unity dynamic range: max(|x[n]|) = 1.0
%   8. Hard bounding to [-1.0, 1.0] against numerical floating-point rounding
%
%   Inputs:
%     x         – Input discrete-time sequence (mono or multi-channel column/row matrix)
%     fs        – Source sampling rate in Hz
%     target_fs – Destination sampling rate in Hz (default: 16 000 Hz)
%     options   – (optional) struct with fields:
%                 .enable_denoise (logical, default: false for clean references to protect formants)
%                 .denoise_gain_floor (double in (0, 1], default: 0.25 conservative)
%
%   Outputs:
%     x_clean   – Preconditioned mono discrete-time sequence [N x 1] at target_fs
%     fs        – Target sampling frequency (16 000 Hz)
%     dsp_info  – Diagnostic struct detailing applied DSP stages, energy, and SNR

arguments
    x (:,:) {mustBeNumeric, mustBeFinite}
    fs (1,1) {mustBePositive, mustBeFinite}
    target_fs (1,1) {mustBePositive, mustBeFinite} = 16000
    options.enable_denoise logical = false
    options.denoise_gain_floor double = 0.25
end

if isempty(x)
    error("preprocess_signal:EmptyInput", "Input discrete-time audio sequence cannot be empty.");
end

dsp_info = struct();
dsp_info.originalFs = fs;
dsp_info.targetFs = target_fs;
dsp_info.resampled = false;
dsp_info.dcRemoved = false;
dsp_info.denoiseApplied = false;

% --- 1. Multi-Channel Spatial Down-Mixing -------------------------------
x = double(x);
if ~isvector(x)
    x = mean(x, 2); % Spatial average across columns
else
    x = x(:);       % Ensure column vector orientation [N x 1]
end

% --- 2. Nyquist–Shannon Anti-Aliasing Resampling ------------------------
% To prevent spectral aliasing when converting between sampling domains,
% apply polyphase rational interpolation/decimation with a Kaiser/FIR anti-aliasing filter.
if abs(fs - target_fs) > 1
    [p, q] = rat(target_fs / fs, 1e-4);
    x = resample(x, p, q);
    dsp_info.resampled = true;
    dsp_info.resampleRatio = [p, q];
end
fs = target_fs;

% --- 3. DC Bias / Mean Removal ------------------------------------------
% Eliminates recording hardware DC offsets that would distort Fourier spectrum at 0 Hz
dc_offset = mean(x);
x = x - dc_offset;
dsp_info.dcRemoved = true;
dsp_info.dcOffset = dc_offset;

% --- 4. Conservative Spectral-Domain Noise Attenuation ------------------
% Crucial Voice-Cloning Protection:
% Do NOT blindly assume t in [0, 0.5] s is silent noise! If the speaker begins
% speaking at t=0, treating [0, 0.5] s as noise will subtract the speaker's
% own formant frequencies, severely degrading vocal naturalness and timbre.
if options.enable_denoise
    [x, dsp_info.denoiseApplied] = conservative_spectral_gate(x, fs, options.denoise_gain_floor);
else
    dsp_info.denoiseApplied = false;
end

% --- 5. Peak Normalization and Dynamic Range Bounding -------------------
peak = max(abs(x), [], "all");
dsp_info.preNormPeak = peak;

if peak > eps
    x_clean = x ./ peak;
else
    % Silent audio sequence
    x_clean = zeros(size(x));
    dsp_info.postNormPeak = 0;
    return;
end

% Guard against float precision overflow
x_clean = max(-1.0, min(1.0, x_clean));
dsp_info.postNormPeak = max(abs(x_clean));
dsp_info.durationSeconds = numel(x_clean) / fs;
dsp_info.sampleCount = numel(x_clean);

end

function [y, applied] = conservative_spectral_gate(x, fs, gainFloor)
% CONSERVATIVE_SPECTRAL_GATE
% Identifies minimum energy frames across the signal to form an accurate
% stationary noise profile without corrupting initial speech formants.
applied = false;
y = x;

N = 1024;    % Frame length
hop = 256;   % Hop size (75% overlap)
win = hann(N, "periodic");

if numel(x) < 2 * N
    % Signal too short for reliable spectral analysis
    return;
end

% 1. STFT decomposition
[S, ~, ~] = stft(x, fs, Window=win, OverlapLength=N-hop, FFTLength=N);
mag = abs(S);
phi = angle(S);
nFrames = size(mag, 2);

if nFrames < 4
    return;
end

% 2. Energy-guided noise floor estimation:
% Find 15% lowest-energy frames across the utterance instead of assuming t=0 is silence
frame_energy = sum(mag.^2, 1);
[~, sort_order] = sort(frame_energy, 'ascend');
nNoiseFrames = max(2, round(0.15 * nFrames));
noise_frame_indices = sort_order(1:nNoiseFrames);

noise_floor = median(mag(:, noise_frame_indices), 2);

% 3. Spectral Wiener-like gain calculation with conservative floor
threshold = max(noise_floor * 1.5, eps);
gain = max(gainFloor, min(1.0, mag ./ threshold));

% 4. Frequency-domain reconstruction & synthesis
S_clean = gain .* mag .* exp(1i .* phi);
y_synth = istft(S_clean, fs, Window=win, OverlapLength=N-hop, FFTLength=N);
y = y_synth(:);

% Align output length exactly with input
if numel(y) > numel(x)
    y = y(1:numel(x));
elseif numel(y) < numel(x)
    y(end+1:numel(x), 1) = 0;
end

applied = true;
end
