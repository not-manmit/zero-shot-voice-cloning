function test_end_to_end_real_reference()
%TEST_END_TO_END_REAL_REFERENCE Real human speech zero-shot voice-cloning validation.
%
%   Execution Protocol:
%     1. Resolves reference speech audio provided by the human developer:
%        - Checks environment variable: getenv('TEST_REFERENCE_AUDIO')
%        - Checks canonical fixture file: tests/reference_speech.wav
%        - Checks root fixture: reference_speech.wav
%     2. IF AUDIO IS NOT FOUND:
%        Gracefully reports SKIPPED (does NOT fake a pass).
%     3. IF AUDIO IS FOUND:
%        - Reads real human audio (mono or stereo, any sampling rate)
%        - Preconditions via preprocess_signal (16 kHz mono conversion)
%        - Extracts 512-dim CAM++ speaker embedding
%        - Synthesizes target English sentence via SpeechT5 + HiFi-GAN
%        - Validates waveform finiteness, duration, and amplitude bounds
%        - Saves generated clone to outputs/real_reference_cloned.wav

fprintf("[test_end_to_end_real_reference] Checking for human speech reference audio ...\n");

cfg = pipeline_config();

% 1. Search for human reference audio
test_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(test_dir);

candidate_paths = {
    getenv('TEST_REFERENCE_AUDIO'), ...
    fullfile(test_dir, 'reference_speech.wav'), ...
    fullfile(repo_root, 'reference_speech.wav')
};

ref_path = "";
for i = 1:numel(candidate_paths)
    cand = strtrim(string(candidate_paths{i}));
    if strlength(cand) > 0 && isfile(cand)
        ref_path = cand;
        break;
    end
end

if strlength(ref_path) == 0
    fprintf("\n=================================================================\n");
    fprintf("[test_end_to_end_real_reference] REAL REFERENCE TEST SKIPPED\n");
    fprintf("Reason: Developer speech audio fixture was not supplied.\n");
    fprintf("To run real voice cloning validation in MATLAB Online:\n");
    fprintf("  1. Upload a 3–5 second clean WAV speech file (any sample rate)\n");
    fprintf("  2. Either name it 'tests/reference_speech.wav'\n");
    fprintf("     OR set: setenv('TEST_REFERENCE_AUDIO', 'path/to/speech.wav')\n");
    fprintf("  3. Run: run('tests/test_end_to_end_real_reference.m')\n");
    fprintf("=================================================================\n\n");
    return; % Gracefully return without asserting false success
end

fprintf("  Found reference audio: %s\n", ref_path);

% 2. Verify Models are Present
requiredModels = {cfg.paths.encoder, cfg.paths.decoder, cfg.paths.decoder_kv, cfg.paths.vocoder, cfg.paths.spk_encoder};
for i = 1:numel(requiredModels)
    if ~isfile(requiredModels{i})
        error("test_end_to_end_real_reference:MissingModel", ...
            "Model asset is missing: %s. Run scripts/setup_matlab_online.m.", requiredModels{i});
    end
end

models = load_onnx_engine(cfg);

% 3. Read and Validate Human Reference Audio
[x_raw, fs_raw] = audioread(ref_path);
dur_raw = numel(x_raw(:, 1)) / fs_raw;
fprintf("  Audio properties: %.2f s, %d Hz, %d channel(s)\n", dur_raw, fs_raw, size(x_raw, 2));

if dur_raw < cfg.spk_min_dur_s
    error("test_end_to_end_real_reference:AudioTooShort", ...
        "Reference audio duration (%.2f s) is below minimum %.1f s threshold.", dur_raw, cfg.spk_min_dur_s);
end

target_text = "This voice was synthesized using real human speaker conditioning in MATLAB Online.";

% 4. Run Complete Voice-Cloning Pipeline
t_gen = tic;
result = generate_voice(x_raw, fs_raw, target_text, models, cfg);
gen_time = toc(t_gen);

% 5. Validate Cloned Output
assert(~isempty(result.waveform), "Generated cloned waveform is empty");
assert(all(isfinite(result.waveform)), "Generated waveform contains non-finite values");
assert(result.sampleRate == cfg.fs, "Sample rate mismatch");
assert(numel(result.waveform) > cfg.fs * 0.5, "Generated audio duration is unexpectedly brief");
assert(max(abs(result.waveform)) <= 1.0 + eps, "Waveform amplitude exceeds [-1, 1]");
assert(numel(result.speakerEmbedding) == cfg.spk_emb_dim, "Speaker embedding dimension mismatch");
assert(abs(norm(result.speakerEmbedding) - 1.0) < 1e-4, "Speaker embedding not L2 normalized");

% 6. Save Result to outputs/
out_dir = fullfile(repo_root, 'outputs');
if ~isfolder(out_dir)
    mkdir(out_dir);
end
dest_wav = fullfile(out_dir, 'real_reference_cloned.wav');
audiowrite(dest_wav, result.waveform, result.sampleRate);

fprintf("\n=================================================================\n");
fprintf("[test_end_to_end_real_reference] REAL SPEECH VOICE CLONING SUCCESS!\n");
fprintf("  Reference source:     %s\n", ref_path);
fprintf("  Target text:          '%s'\n", target_text);
fprintf("  Generated duration:   %.2f s (%d samples at %d Hz)\n", ...
    numel(result.waveform)/result.sampleRate, numel(result.waveform), result.sampleRate);
fprintf("  Total synthesis time: %.2f s (RTF: %.2fx)\n", ...
    gen_time, gen_time / (numel(result.waveform)/result.sampleRate));
fprintf("  Output WAV written:   %s\n", dest_wav);
fprintf("=================================================================\n");

end
