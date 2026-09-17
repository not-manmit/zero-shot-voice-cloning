function [log_mel, S, f, t] = stft_analysis(x, fs, cfg)
%STFT_ANALYSIS  Hann-windowed STFT and log-Mel spectro-temporal matrix.
%
%   [log_mel, S, f, t] = STFT_ANALYSIS(x, fs)
%   [log_mel, S, f, t] = STFT_ANALYSIS(x, fs, cfg)
%
%   Computes the short-time Fourier transform (STFT) of the discrete-time
%   sequence x sampled at fs Hz, then projects the power spectrum onto an
%   80-band HTK Mel filter-bank and converts to a log scale.
%
%   The resulting log_mel matrix is compatible with the SpeechT5
%   FeatureExtractor defaults used during model pre-training:
%
%       sampling_rate : 16 000 Hz
%       n_fft         : 1 024  samples   (N-point DFT)
%       hop_length    :   256  samples   (16 ms)
%       win_length    : 1 024  samples   (64 ms)
%       window        : Hann  (periodic)
%       n_mels        :    80
%       f_min         :    80 Hz
%       f_max         : 7 600 Hz
%       mel_floor     : 1e-10
%       log_base      : natural log   (torch.log, not log10 / dB)
%       normalise     : none
%
%   Inputs
%   ------
%   x   – column vector, mono audio at fs Hz (double or single)
%   fs  – scalar sample rate (Hz); must equal 16 000 for SpeechT5
%   cfg – (optional) struct from pipeline_config(); uses defaults if absent
%
%   Outputs
%   -------
%   log_mel  – [n_mels × T_frames]  float32  log-Mel matrix
%   S        – [N/2+1 × T_frames]   complex  STFT (one-sided)
%   f        – [N/2+1 × 1]          frequency axis (Hz)
%   t        – [1 × T_frames]       time axis (s)

arguments
    x   (:,1) {mustBeNumeric, mustBeFinite}
    fs  (1,1) {mustBePositive, mustBeFinite}
    cfg struct = struct()   % populated below if absent
end

% -----------------------------------------------------------------------
% Resolve configuration
% -----------------------------------------------------------------------
if ~isfield(cfg, 'n_fft')
    default_cfg = pipeline_config();
    fields = fieldnames(default_cfg);
    for k = 1:numel(fields)
        if ~isfield(cfg, fields{k})
            cfg.(fields{k}) = default_cfg.(fields{k});
        end
    end
end

N   = cfg.n_fft;          % DFT length  (1024)
hop = cfg.hop_length;     % frame step  (256 samples)
win = hann(N, "periodic");

% Warn if sample rate disagrees with the expected model convention.
if abs(fs - cfg.fs) > 1
    warning("stft_analysis:SampleRateMismatch", ...
        "Input sample rate %d Hz differs from model convention %d Hz. " + ...
        "Call preprocess_signal first.", fs, cfg.fs);
end

% -----------------------------------------------------------------------
% Short-Time Fourier Transform
%   stft() returns the two-sided spectrum; we keep the one-sided form.
% -----------------------------------------------------------------------
x_d = double(x);
[S_full, f_full, t] = stft(x_d, fs, ...
    Window=win, OverlapLength=N-hop, FFTLength=N, ...
    FrequencyRange="onesided");

S = S_full;
f = f_full;

% -----------------------------------------------------------------------
% Power spectrum  (|STFT|²)
% -----------------------------------------------------------------------
power = real(S .* conj(S));    % [N/2+1 × T_frames]

% -----------------------------------------------------------------------
% HTK Mel filter-bank
%   Triangular filters uniformly spaced in the HTK Mel scale:
%       mel = 2595 · log10(1 + f / 700)
%   f_min and f_max are taken from the SpeechT5 FeatureExtractor config.
% -----------------------------------------------------------------------
log_mel = apply_mel_filterbank(power, f, cfg);   % [n_mels × T_frames]

% -----------------------------------------------------------------------
% Log compression  – natural logarithm, matching torch.log behaviour.
%   log_mel = log(max(filter_output, mel_floor))
% -----------------------------------------------------------------------
log_mel = log(max(log_mel, cfg.mel_floor));
log_mel = single(log_mel);    % store as float32 (matches model dtype)
end


% -----------------------------------------------------------------------
% Local helper: HTK Mel filter-bank
% -----------------------------------------------------------------------
function mel_out = apply_mel_filterbank(power, f, cfg)
%APPLY_MEL_FILTERBANK Project linear power spectrum onto Mel bands.
%
%   Each triangular filter is defined by (left, centre, right) in Hz,
%   linearly interpolated in the HTK Mel scale.
%
%   Mel scale:  mel = 2595 · log10(1 + f/700)
%               f   = 700 · (10^(mel/2595) – 1)

n_mels = cfg.n_mels;
f_min  = cfg.f_min;
f_max  = cfg.f_max;

% Mel-frequency centre-points for n_mels+2 pivot frequencies
mel_min = hz2mel(f_min);
mel_max = hz2mel(f_max);
mel_pts = linspace(mel_min, mel_max, n_mels + 2);   % n_mels+2 pivots
hz_pts  = mel2hz(mel_pts);                           % back to Hz

% Map pivot Hz to nearest FFT bin
n_bins  = numel(f);                                  % N/2+1
bin_pts = zeros(1, n_mels + 2);
for k = 1:n_mels + 2
    [~, bin_pts(k)] = min(abs(f - hz_pts(k)));
end

% Build and apply each triangular filter
mel_out = zeros(n_mels, size(power, 2), "double");
for m = 1:n_mels
    b_left   = bin_pts(m);
    b_centre = bin_pts(m + 1);
    b_right  = bin_pts(m + 2);

    % Clamp to valid range
    b_left   = max(1, b_left);
    b_right  = min(n_bins, b_right);
    b_centre = max(b_left + 1, min(b_right - 1, b_centre));

    % Rising slope  [b_left … b_centre]
    n_rise = b_centre - b_left + 1;
    if n_rise >= 1
        rise = linspace(0, 1, n_rise)';        % column vector
        mel_out(m, :) = mel_out(m, :) + rise' * power(b_left:b_centre, :);
    end

    % Falling slope  (b_centre … b_right]  (exclude centre to avoid ×2)
    n_fall = b_right - b_centre;
    if n_fall >= 1
        fall = linspace(1, 0, n_fall + 1)';    % includes centre
        fall = fall(2:end);                     % drop centre
        mel_out(m, :) = mel_out(m, :) + fall' * power(b_centre+1:b_right, :);
    end
end
end


% -----------------------------------------------------------------------
% HTK Mel ↔ Hz conversion
% -----------------------------------------------------------------------
function m = hz2mel(f)
    m = 2595 .* log10(1 + f ./ 700);
end

function f = mel2hz(m)
    f = 700 .* (10 .^ (m ./ 2595) - 1);
end
