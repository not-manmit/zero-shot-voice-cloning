function [matched_wave, shift_ratio] = match_speaker_acoustics(syn_wave, target_f0, fs, options)
%MATCH_SPEAKER_ACOUSTICS Classical DSP pitch and vocal tract post-processor.
%
%   [matched_wave, shift_ratio] = MATCH_SPEAKER_ACOUSTICS(syn_wave, target_f0, fs)
%   [matched_wave, shift_ratio] = MATCH_SPEAKER_ACOUSTICS(syn_wave, target_f0, fs, options)
%
%   Signals & Systems DSP Post-Processor:
%     Applies classical pitch modification (resampling + WSOLA time-scale
%     modification) to adjust synthesized speech F0 toward the reference
%     speaker's measured fundamental frequency, preserving speech rate
%     and natural formant structure.
%
%   Inputs:
%     syn_wave  – Synthesized speech sequence [N x 1]
%     target_f0 – Measured reference speaker median pitch (Hz) in [60, 350]
%     fs        – Sampling rate (Hz), default 16000
%     options   – struct with optional fields:
%                 .max_shift_semitones – max allowed shift (default: 4.0)
%                 .blend_factor        – blend toward target (default: 0.6)
%
%   Outputs:
%     matched_wave – Pitch-matched waveform [N x 1]
%     shift_ratio  – Applied pitch shift multiplier (e.g. 1.05 = +5% pitch)

arguments
    syn_wave  (:,1) {mustBeNumeric, mustBeFinite}
    target_f0 (1,1) {mustBeNumeric, mustBeFinite} = 0
    fs        (1,1) {mustBePositive, mustBeFinite} = 16000
    options.max_shift_semitones (1,1) {mustBePositive} = 4.0
    options.blend_factor        (1,1) {mustBeInRange(options.blend_factor, 0.0, 1.0)} = 0.6
end

matched_wave = double(syn_wave(:));
shift_ratio = 1.0;

if isempty(matched_wave) || target_f0 < 60 || target_f0 > 350
    return;
end

% 1. Estimate current F0 of synthesized speech via autocorrelation
syn_f0 = estimate_f0(matched_wave, fs);
if syn_f0 < 60 || syn_f0 > 350
    return;
end

% 2. Compute pitch shift ratio with blending and safe bounding
raw_ratio = target_f0 / syn_f0;
% Apply conservative blend factor
effective_ratio = 1.0 + options.blend_factor * (raw_ratio - 1.0);

% Bound shift to +/- max_shift_semitones
max_ratio = 2 ^ (options.max_shift_semitones / 12);
min_ratio = 2 ^ (-options.max_shift_semitones / 12);
shift_ratio = max(min_ratio, min(max_ratio, effective_ratio));

% If shift is negligible (< 2%), skip processing
if abs(shift_ratio - 1.0) < 0.02
    shift_ratio = 1.0;
    return;
end

% 3. Pitch shifting via classical resampling + WSOLA time-stretching
% Step A: Resample by shift_ratio to shift pitch (and inadvertently change length)
% Step B: Time-stretch by shift_ratio to restore original duration
try
    % Rational approximation of shift_ratio for resample
    [P, Q] = rat(shift_ratio, 1e-3);
    resampled = resample(matched_wave, P, Q);

    % WSOLA time-scale modification to restore original duration
    target_len = numel(matched_wave);
    stretched = wsola_timescale(resampled, target_len, fs);

    if numel(stretched) == target_len && all(isfinite(stretched))
        matched_wave = stretched;
    elseif numel(stretched) > 0 && all(isfinite(stretched))
        % Length alignment
        if numel(stretched) > target_len
            matched_wave = stretched(1:target_len);
        else
            matched_wave = [stretched; zeros(target_len - numel(stretched), 1)];
        end
    end

    % Safe peak normalization
    pk = max(abs(matched_wave));
    if pk > 1.0
        matched_wave = matched_wave / pk;
    end

    fprintf("[match_speaker_acoustics] Adjusted pitch: syn=%.1f Hz -> target=%.1f Hz (shift: %.2fx)\n", ...
        syn_f0, target_f0, shift_ratio);
catch ME
    % Fall back to original waveform safely on any DSP error
    warning("match_speaker_acoustics:DspFailed", ...
        "Acoustic post-processing failed (%s); keeping original waveform.", ME.message);
    matched_wave = double(syn_wave(:));
    shift_ratio = 1.0;
end

end

function f0 = estimate_f0(x, fs)
% Simple autocorrelation pitch estimator for synthesized speech
x_pre = filter([1, -0.97], 1, double(x));
frame_len = round(0.030 * fs);
hop = round(0.015 * fs);
min_lag = round(fs / 350);
max_lag = round(fs / 60);

n_frames = max(1, floor((numel(x_pre) - frame_len) / hop) + 1);
f0_candidates = [];
win = hann(frame_len, 'periodic');

for k = 1:min(50, n_frames)
    s = (k-1) * hop + 1;
    e = min(s + frame_len - 1, numel(x_pre));
    if e - s + 1 < frame_len, break; end
    frm = x_pre(s:e) .* win;
    if sum(frm.^2) < 1e-5, continue; end
    r = xcorr(frm, max_lag, 'coeff');
    r = r(max_lag+1:end);
    if max_lag <= min_lag, continue; end
    [pk, pk_idx] = max(r(min_lag:min(numel(r), max_lag)));
    lag = min_lag + pk_idx - 1;
    if pk > 0.3 && lag > 0
        f0_candidates(end+1) = fs / lag; %#ok<AGROW>
    end
end

if isempty(f0_candidates)
    f0 = 130;
else
    f0 = median(f0_candidates);
end
end

function y = wsola_timescale(x, target_len, fs)
%WSOLA Waveform Similarity Overlap-Add time-scale modification
N_in = numel(x);
if N_in == 0 || target_len == 0
    y = x;
    return;
end

alpha = target_len / N_in;  % Time-stretch factor (>1 = slower, <1 = faster)
if abs(alpha - 1.0) < 0.01
    y = x;
    return;
end

win_len = round(0.025 * fs);  % 25 ms window
hop_synth = round(win_len / 2);
hop_analysis = round(hop_synth / alpha);
max_delta = round(hop_synth / 2);

win = hann(win_len, 'periodic');
y = zeros(target_len, 1);
norm_weight = zeros(target_len, 1);

synth_pos = 1;
ana_pos = 1;

while synth_pos + win_len - 1 <= target_len && ana_pos + win_len - 1 + max_delta <= N_in
    % Search for best alignment in [ana_pos - max_delta, ana_pos + max_delta]
    best_delta = 0;
    if synth_pos > 1
        ref_seg = y(synth_pos : min(synth_pos + win_len - 1, target_len));
        best_corr = -inf;
        for delta = -max_delta:max_delta
            cand_pos = ana_pos + delta;
            if cand_pos < 1 || cand_pos + numel(ref_seg) - 1 > N_in, continue; end
            cand_seg = x(cand_pos : cand_pos + numel(ref_seg) - 1);
            c = sum(ref_seg .* cand_seg);
            if c > best_corr
                best_corr = c;
                best_delta = delta;
            end
        end
    end

    opt_ana = max(1, min(N_in - win_len + 1, ana_pos + best_delta));
    frm = x(opt_ana : opt_ana + win_len - 1) .* win;

    y(synth_pos : synth_pos + win_len - 1) = y(synth_pos : synth_pos + win_len - 1) + frm;
    norm_weight(synth_pos : synth_pos + win_len - 1) = norm_weight(synth_pos : synth_pos + win_len - 1) + win.^2;

    synth_pos = synth_pos + hop_synth;
    ana_pos = ana_pos + hop_analysis;
end

% Normalize overlap-add weights
valid = norm_weight > 1e-4;
y(valid) = y(valid) ./ norm_weight(valid);
end
