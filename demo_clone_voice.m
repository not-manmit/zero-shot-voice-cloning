%DEMO_CLONE_VOICE One-click zero-shot voice cloning execution in MATLAB Online.
%
%   Run this script directly in the MATLAB Online Command Window:
%       demo_clone_voice
%
%   Workflow:
%     1. Loads all 5 ONNX neural network models into memory.
%     2. Discovers or creates a reference speaker audio sample.
%     3. Synthesizes new speech conditioned on that speaker's voice.
%     4. Plays the cloned speech audio.
%     5. Saves the output audio file to outputs/cloned_voice.wav.

fprintf("\n=================================================================\n");
fprintf("      ZERO-SHOT VOICE CLONING DEMO (MATLAB ONLINE EDITION)       \n");
fprintf("=================================================================\n\n");

% 1. Add repository paths
projRoot = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(projRoot, 'src')));
addpath(fullfile(projRoot, 'scripts')));
addpath(fullfile(projRoot, 'ui'));

cfg = pipeline_config();

% 2. Load Models
fprintf("[1/4] Loading ONNX neural networks into session cache ...\n");
models = load_onnx_engine(cfg, true);

% 3. Prepare Reference Speaker Voice
fprintf("\n[2/4] Preparing reference voice ...\n");
refFile = fullfile(projRoot, 'tests', 'reference_speech.wav');

if isfile(refFile)
    [refAudio, refFs] = audioread(refFile);
    fprintf("  Using reference audio fixture: %s (%.2f s, %d Hz)\n", ...
        refFile, numel(refAudio)/refFs, refFs);
else
    % Synthesize a rich harmonic reference signal representing vocal resonance
    fprintf("  [NOTICE] No reference WAV found at 'tests/reference_speech.wav'.\n");
    fprintf("  Synthesizing acoustic reference voice signal (you can upload your own WAV anytime!).\n");
    refFs = 16000;
    t = (0:1/refFs:2.5-1/refFs)';
    % Fundamental frequency ~130 Hz with vocal tract formant harmonics
    f0 = 130;
    refAudio = 0.4 * sin(2*pi*f0*t) + ...
               0.3 * sin(2*pi*f0*2*t) + ...
               0.2 * sin(2*pi*f0*3*t) + ...
               0.1 * sin(2*pi*f0*5*t) + ...
               0.05 * randn(size(t));
    refAudio = refAudio / max(abs(refAudio));
end

% 4. Target Synthesis Sentence
targetText = "Hello! This is a zero-shot voice cloning demonstration running entirely in MATLAB.";
fprintf("\n[3/4] Synthesizing speech conditioned on reference voice:\n");
fprintf("  Target Text: '%s'\n\n", targetText);

t_start = tic;
result = generate_voice(refAudio, refFs, targetText, models, cfg);
elapsed = toc(t_start);

% 5. Save & Play Output
outDir = fullfile(projRoot, 'outputs');
if ~isfolder(outDir)
    mkdir(outDir);
end
outFile = fullfile(outDir, 'cloned_voice.wav');
audiowrite(outFile, result.waveform, result.sampleRate);

fprintf("\n[4/4] Voice Cloning Complete!\n");
fprintf("  Synthesized audio duration: %.2f s (%d samples at %d Hz)\n", ...
    result.metadata.audioDurationSeconds, numel(result.waveform), result.sampleRate);
fprintf("  Total synthesis time:       %.2f s (RTF: %.2fx)\n", ...
    elapsed, result.metadata.realTimeFactor);
fprintf("  Saved WAV file:             %s\n", outFile);

% Attempt audio playback in MATLAB Online
try
    sound(result.waveform, result.sampleRate);
    fprintf("  Audio playback started!\n");
catch
    fprintf("  (To listen, double-click '%s' in the Files panel on the left).\n", outFile);
end

fprintf("\n=================================================================\n");
fprintf("To launch the full interactive Graphical User Interface with\n");
fprintf("microphone recording, file upload, and waveform/Mel displays, run:\n");
fprintf("  VoiceClonerApp\n");
fprintf("=================================================================\n\n");
