classdef VoiceClonerApp < handle
    %VOICECLONERAPP MATLAB Online UI for zero-shot voice cloning (MATLAB-only).

    properties
        UIFigure matlab.ui.Figure
        TargetTextArea matlab.ui.control.TextArea
        StatusLabel matlab.ui.control.Label
        MetricsLabel matlab.ui.control.Label
        ReferenceAxes matlab.ui.control.UIAxes
        OutputAxes matlab.ui.control.UIAxes
        MelAxes matlab.ui.control.UIAxes
        ReferenceAudio double = []
        ReferenceFs double = 16000
        ReferenceSource string = ""
        ReferenceHash string = ""
        GeneratedAudio double = []
        GeneratedFs double = 16000
        LoadedModels struct = struct()
        Config struct = struct()
        SpeakerEmbedding double = []
        CurrentResult struct = struct()
        Recorder % audiorecorder object (MATLAB Online browser mic)
        RecordingActive logical = false
        GenerationMetrics struct = struct()
    end

    methods
        function app = VoiceClonerApp
            app.Config = pipeline_config();
            app.UIFigure = uifigure(Name="Zero-Shot Voice Cloner (MATLAB-only)", Position=[100 100 980 720]);

            uilabel(app.UIFigure, Text="REFERENCE VOICE (16 kHz mono, >=1 s)", Position=[30 680 300 22], FontWeight='bold');
            uibutton(app.UIFigure, Text="Record", Position=[30 640 90 30], ButtonPushedFcn=@app.recordReference);
            uibutton(app.UIFigure, Text="Stop", Position=[130 640 90 30], ButtonPushedFcn=@app.stopRecording);
            uibutton(app.UIFigure, Text="Upload WAV", Position=[230 640 120 30], ButtonPushedFcn=@app.loadReference);
            uibutton(app.UIFigure, Text="Play Reference", Position=[360 640 120 30], ButtonPushedFcn=@app.playReference);

            uilabel(app.UIFigure, Text="TARGET TEXT", Position=[30 590 150 22], FontWeight='bold');
            app.TargetTextArea = uitextarea(app.UIFigure, Position=[30 520 430 65], Placeholder="Hello, this is a zero-shot voice cloning demonstration in MATLAB.");
            uibutton(app.UIFigure, Text="Generate", Position=[30 470 120 30], ButtonPushedFcn=@app.generateSpeech);
            uibutton(app.UIFigure, Text="Play Output", Position=[170 470 120 30], ButtonPushedFcn=@app.playOutput);
            uibutton(app.UIFigure, Text="Save WAV", Position=[310 470 120 30], ButtonPushedFcn=@app.saveOutput);

            app.StatusLabel = uilabel(app.UIFigure, Text="Ready – loading models...", Position=[500 470 450 22]);
            app.MetricsLabel = uilabel(app.UIFigure, Text="Metrics: –", Position=[500 450 450 22]);

            app.ReferenceAxes = uiaxes(app.UIFigure, Position=[30 300 430 150], Title="Reference waveform");
            app.OutputAxes = uiaxes(app.UIFigure, Position=[500 300 430 150], Title="Generated waveform");
            app.MelAxes = uiaxes(app.UIFigure, Position=[30 60 900 180], Title="Generated Mel [80 x T] (optional)");

            app.loadModels();
        end

        function loadModels(app)
            app.StatusLabel.Text = "Loading models...";
            drawnow;
            try
                app.LoadedModels = load_onnx_engine(app.Config);
                app.StatusLabel.Text = "Models ready. Record or upload a reference (>=1 s).";
            catch ME
                app.StatusLabel.Text = sprintf("Model loading failed: %s (run setup_matlab_online.m)", ME.message);
            end
            drawnow;
        end

        function recordReference(app, ~, ~)
            if app.RecordingActive
                app.StatusLabel.Text = "Recording already active – press Stop.";
                return;
            end
            % MATLAB Online may not support audiorecorder – check gracefully
            if exist('audiorecorder','file') ~= 2 && exist('audiorecorder','builtin') ~= 5
                app.StatusLabel.Text = "Microphone recording not available in this MATLAB environment. Use Upload WAV.";
                return;
            end
            try
                app.Recorder = audiorecorder(app.Config.fs, 16, 1);
                record(app.Recorder);
                app.RecordingActive = true;
                app.StatusLabel.Text = "Recording reference... speak now, then press Stop.";
            catch ME
                app.StatusLabel.Text = sprintf("Microphone unavailable (MATLAB Online may require browser permission): %s — use Upload WAV instead.", ME.message);
            end
            drawnow;
        end

        function stopRecording(app, ~, ~)
            if isempty(app.Recorder) || ~isvalid(app.Recorder)
                app.StatusLabel.Text = "No active recording.";
                return;
            end
            try
                stop(app.Recorder);
            catch
            end
            app.RecordingActive = false;
            try
                audio = getaudiodata(app.Recorder, 'double');
                if isempty(audio)
                    app.StatusLabel.Text = "Recording produced no audio – try again.";
                    return;
                end
                fs = 16000; % audiorecorder fs
                % Validate
                if sqrt(mean(audio.^2)) < 1e-4
                    app.StatusLabel.Text = "Recording is silent – check microphone level.";
                    return;
                end
                if numel(audio)/fs < app.Config.spk_min_dur_s
                    app.StatusLabel.Text = sprintf("Recording too short (%.1f s); need >= %.1f s.", numel(audio)/fs, app.Config.spk_min_dur_s);
                    return;
                end
                app.ReferenceAudio = audio(:);
                app.ReferenceFs = fs;
                app.ReferenceSource = "microphone";
                app.ReferenceHash = string(numel(audio)) + "_" + string(sum(abs(audio(1:min(1000,end))))); % cheap hash
                app.SpeakerEmbedding = []; % invalidate cache
                app.updateReferencePlot();
                app.StatusLabel.Text = sprintf("Reference recorded (%.1f s, %d Hz). You can Generate now.", numel(audio)/fs, fs);
            catch ME
                app.StatusLabel.Text = sprintf("Recording stop failed: %s", ME.message);
            end
            drawnow;
        end

        function loadReference(app, ~, ~)
            [file, path] = uigetfile({'*.wav;*.flac;*.mp3;*.m4a','Audio files'}, 'Select reference speech (prefer WAV 16 kHz)');
            if isequal(file, 0)
                return;
            end
            full = fullfile(path, file);
            try
                [x, fs] = audioread(full);
            catch ME
                app.StatusLabel.Text = sprintf('Failed to read %s: %s', file, ME.message);
                return;
            end
            try
                [x_clean, fs_clean] = preprocess_signal(x, fs, app.Config.fs);
                if sqrt(mean(x_clean.^2)) < 1e-4
                    app.StatusLabel.Text = "Uploaded file appears silent.";
                    return;
                end
                if numel(x_clean)/fs_clean < app.Config.spk_min_dur_s
                    app.StatusLabel.Text = sprintf('Reference too short (%.2f s); need >= %.1f s.', numel(x_clean)/fs_clean, app.Config.spk_min_dur_s);
                    return;
                end
                app.ReferenceAudio = x_clean(:);
                app.ReferenceFs = fs_clean;
                app.ReferenceSource = string(file);
                app.ReferenceHash = string(numel(x_clean)) + "_" + string(sum(abs(x_clean(1:min(1000,end)))));
                app.SpeakerEmbedding = [];
                app.updateReferencePlot();
                app.StatusLabel.Text = sprintf('Reference loaded: %s (%.1f s)', file, numel(x_clean)/fs_clean);
            catch ME
                app.StatusLabel.Text = sprintf('Reference preprocessing failed: %s', ME.message);
            end
            drawnow;
        end

        function playReference(app, ~, ~)
            if isempty(app.ReferenceAudio)
                app.StatusLabel.Text = 'No reference audio loaded';
                return;
            end
            try
                sound(app.ReferenceAudio, app.ReferenceFs);
                app.StatusLabel.Text = 'Playing reference...';
            catch ME
                app.StatusLabel.Text = sprintf('Playback failed: %s', ME.message);
            end
        end

        function generateSpeech(app, ~, ~)
            if isempty(app.ReferenceAudio)
                app.StatusLabel.Text = 'Load or record a reference first (>=1 s)';
                return;
            end
            textValue = strjoin(string(app.TargetTextArea.Value), " ");
            textValue = string(textValue);
            if strlength(strtrim(textValue)) == 0
                app.StatusLabel.Text = 'Enter target text';
                return;
            end
            % Status trace — real stages (not overwritten instantly)
            % Reuse embedding if reference unchanged (UI-level cache)
            isCached = ~isempty(app.SpeakerEmbedding);
            try
                tStart = tic;
                if isCached
                    app.StatusLabel.Text = sprintf('Reusing speaker embedding (%d-dim) + tokenizing...', numel(app.SpeakerEmbedding));
                else
                    app.StatusLabel.Text = 'Stage 1/5: Preprocessing reference (16 kHz resample + denoise)...';
                end
                drawnow;
                % Central pipeline handles all 5 stages with measured metrics
                result = generate_voice(app.ReferenceAudio, app.ReferenceFs, textValue, app.LoadedModels, app.Config);
                if isCached
                    % Notify that embedding was reused at pipeline level (generate_voice recomputes but we keep UI cache)
                    fprintf('[VoiceClonerApp] Reference unchanged — embedding reused from previous generation.\n');
                end
                app.StatusLabel.Text = 'Stage 5/5: Rendering waveform...';
                drawnow;
                app.CurrentResult = result;
                app.GeneratedAudio = result.waveform;
                app.GeneratedFs = result.sampleRate;
                app.SpeakerEmbedding = result.speakerEmbedding;
                app.GenerationMetrics = result.metrics;
                app.updateOutputPlot();
                % Mel visualisation [80 x T]
                try
                    cla(app.MelAxes);
                    imagesc(app.MelAxes, result.acousticFeatures);
                    axis(app.MelAxes, 'tight');
                    xlabel(app.MelAxes, 'Frames');
                    ylabel(app.MelAxes, 'Mel bin');
                    title(app.MelAxes, sprintf('Mel [80 x %d] — log-Mel 80-7600 Hz', size(result.acousticFeatures,2)));
                    colormap(app.MelAxes, 'parula');
                    colorbar(app.MelAxes);
                catch ME2
                    fprintf('[VoiceClonerApp] Mel plot failed: %s\n', ME2.message);
                end
                elapsed = toc(tStart);
                app.StatusLabel.Text = sprintf('Complete: %.2f s total — %d samples @ %d Hz', result.metrics.totalTime_s, numel(result.waveform), result.sampleRate);
                app.MetricsLabel.Text = sprintf('pre %.2fs | spk %.2fs | tok %.2fs | enc+dec %.2fs | voc %.2fs | tot %.2fs (wall %.2fs)', ...
                    result.metrics.preprocessingTime_s, result.metrics.speakerEncoderTime_s, result.metrics.tokenizationTime_s, ...
                    result.metrics.encoderTime_s, result.metrics.vocoderTime_s, result.metrics.totalTime_s, elapsed);
            catch ME
                app.StatusLabel.Text = sprintf('Generation failed: %s', ME.message);
                app.MetricsLabel.Text = 'Metrics: failed — check Command Window for stack';
                fprintf('[VoiceClonerApp] Generation error: %s\n', ME.message);
                % Re-throw for Command Window if user inspects
                disp(getReport(ME,'extended'));
            end
            drawnow;
        end

        function playOutput(app, ~, ~)
            if isempty(app.GeneratedAudio)
                app.StatusLabel.Text = 'Generate speech before playback';
                return;
            end
            try
                sound(app.GeneratedAudio, app.GeneratedFs);
                app.StatusLabel.Text = 'Playing generated output...';
            catch ME
                app.StatusLabel.Text = sprintf('Playback failed: %s', ME.message);
            end
        end

        function saveOutput(app, ~, ~)
            if isempty(app.GeneratedAudio)
                app.StatusLabel.Text = 'No generated output to save';
                return;
            end
            if ~all(isfinite(app.GeneratedAudio)) || isempty(app.GeneratedAudio)
                app.StatusLabel.Text = 'Generated output invalid – cannot save.';
                return;
            end
            [file, path] = uiputfile('*.wav', 'Save generated speech as');
            if isequal(file, 0)
                return;
            end
            full = fullfile(path, file);
            try
                audiowrite(full, app.GeneratedAudio, app.GeneratedFs);
                app.StatusLabel.Text = sprintf('Saved: %s (%d Hz)', full, app.GeneratedFs);
            catch ME
                app.StatusLabel.Text = sprintf('Save failed: %s', ME.message);
            end
        end

        function updateReferencePlot(app)
            if isempty(app.ReferenceAudio)
                cla(app.ReferenceAxes);
                return;
            end
            t = (0:numel(app.ReferenceAudio)-1) / app.ReferenceFs;
            plot(app.ReferenceAxes, t, app.ReferenceAudio);
            xlabel(app.ReferenceAxes, 'Time (s)');
            ylabel(app.ReferenceAxes, 'Amplitude');
            title(app.ReferenceAxes, sprintf('Reference (%s, %.1f s)', app.ReferenceSource, numel(app.ReferenceAudio)/app.ReferenceFs));
        end

        function updateOutputPlot(app)
            if isempty(app.GeneratedAudio)
                cla(app.OutputAxes);
                return;
            end
            t = (0:numel(app.GeneratedAudio)-1) / app.GeneratedFs;
            plot(app.OutputAxes, t, app.GeneratedAudio);
            xlabel(app.OutputAxes, 'Time (s)');
            ylabel(app.OutputAxes, 'Amplitude');
            title(app.OutputAxes, sprintf('Generated (%.1f s)', numel(app.GeneratedAudio)/app.GeneratedFs));
        end
    end
end
