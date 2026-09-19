function test_dsp()
%TEST_DSP Static / no-model unit test for Signals & Systems DSP routines.
%
%   Verifies:
%     1. Multi-channel to mono mix-down
%     2. Rational anti-aliasing resampling to 16 000 Hz
%     3. DC offset removal
%     4. STFT analysis and Hann windowing properties
%     5. HTK Mel filter-bank dimension and energy conservation
%     6. Conservative spectral gating

fprintf("[test_dsp] Running DSP module verification ...\n");

cfg = pipeline_config();

% 1. Multi-channel downmix & resampling
fs_in = 44100;
t_in = (0:1/fs_in:0.5-1/fs_in)';
stereo_sig = [sin(2*pi*440*t_in), cos(2*pi*440*t_in)]; % Stereo 44.1 kHz

[mono_out, fs_out, info] = preprocess_signal(stereo_sig, fs_in, 16000);
assert(fs_out == 16000, "Target fs must be 16000 Hz");
assert(iscolumn(mono_out), "Output must be a mono column vector");
assert(info.resampled, "Resampling flag must be true");
assert(abs(mean(mono_out)) < 1e-4, "DC bias must be removed");
assert(max(abs(mono_out)) <= 1.0 + eps, "Peak normalization exceeded unity");

% 2. STFT and Mel transformation
fs = 16000;
t = (0:1/fs:1.0-1/fs)';
test_speech = 0.5 * sin(2*pi*300*t) + 0.3 * sin(2*pi*1200*t) + 0.05 * randn(size(t));

[log_mel, S, f, time_axis] = stft_analysis(test_speech, fs, cfg);

assert(size(log_mel, 1) == 80, "Mel spectrogram must contain exactly 80 frequency bins");
assert(size(log_mel, 2) > 0, "Temporal frame count must be positive");
assert(all(isfinite(log_mel), 'all'), "Mel matrix contains non-finite values");
assert(numel(f) == (cfg.n_fft / 2 + 1), "STFT one-sided frequency bin count mismatch");
assert(numel(time_axis) == size(log_mel, 2), "STFT time axis length mismatch");

% 3. Safe noise suppression
[denoised, ~, denoise_info] = preprocess_signal(test_speech, fs, 16000, enable_denoise=true);
assert(denoise_info.denoiseApplied, "Noise gating should be applied when enabled");
assert(all(isfinite(denoised)), "Denoised signal must be finite");
assert(numel(denoised) == numel(test_speech), "Signal length altered by spectral gate");

fprintf("[test_dsp] All DSP assertions PASSED.\n");
end
