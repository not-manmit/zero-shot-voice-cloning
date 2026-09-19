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
        ReferenceFingerprint string = ""
        GeneratedAudio double = []
        GeneratedFs double = 16000
        LoadedModels struct = struct()
        Config struct = struct()
        CachedSpeakerEmbedding single = []
        CurrentResult struct = struct()
        Recorder % audiorecorder object
        RecordingActive logical = false
        GenerationMetrics struct = struct()
    end

    methods
        function app = VoiceClonerApp
            app.Config = pipeline_config();
            app.UIFigure = uifigure(Name="Zero-Shot Voice Cloner (MATLAB Online Edition)", Position=[100 100 980 720]);

            uilabel(app.UIFigure, Text="REFERENCE VOICE (16 kHz mono, >=1 s recommended)", Position=[30 680 350 22], FontWeight='bold');
            uibutton(app.UIFigure, Text="Record", Position=[30 640 90 30], ButtonPushedFcn=@app.recordReference);
            uibutton(app.UIFigure, Text="Stop", Position=[130 640 90 30], ButtonPushedFcn=@app.stopRecording);
            uibutton(app.UIFigure, Text="Upload WAV", Position=[230 640 120 30], ButtonPushedFcn=@app.loadReference);
            uibutton(app.UIFigure, Text="Play Reference", Position=[360 640 120 30], ButtonPushedFcn=@app.playReference);

            uilabel(app.UIFigure, Text="TARGET TEXT", Position=[30 590 150 22], FontWeight='bold');
            app.TargetTextArea = uitextarea(app.UIFigure, Position=[30 520 430 65], ...
                Value="Hello, this is a zero-shot voice cloning demonstration in MATLAB.", ...
                Placeholder="Enter English speech synthesis text here...");
            uibutton(app.UIFigure, Text="Generate", Position=[30 470 120 30], ButtonPushedFcn=@app.generateSpeech);
            uibutton(app.UIFigure, Text="Play Output", Position=[170 470 120 30], ButtonPushedFcn=@app.playOutput);
            uibutton(app.UIFigure, Text="Save WAV", Position=[310 470 120 30], ButtonPushedFcn=@app.saveOutput);

            app.StatusLabel = uilabel(app.UIFigure, Text="Initializing – loading ONNX models...", Position=[500 470 460 22]);
            app.MetricsLabel = uilabel(app.UIFigure, Text="Metrics: –", Position=[500 445 460 22]);

            app.ReferenceAxes = uiaxes(app.UIFigure, Position=[30 290 430 140]);
            title(app.ReferenceAxes, "Reference Speech Waveform");

            app.OutputAxes = uiaxes(app.UIFigure, Position=[500 290 430 140]);
            title(app.OutputAxes, "Cloned Speech Waveform");

            app.MelAxes = uiaxes(app.UIFigure, Position=[30 60 900 190]);
            title(app.MelAxes, "Synthesized 80-bin Mel Spectrogram");

            app.loadModels();
        end

        function loadModels(app)
            app.StatusLabel.Text = "Loading ONNX models into persistent cache...";
            drawnow;
            try
                app.LoadedModels = load_onnx_engine(app.Config);
                app.StatusLabel.Text = "Models ready. Upload or record reference speech (>=1.0 s).";
            catch ME
                app.StatusLabel.Text = sprintf("Model load failed: %s (run scripts/setup_matlab_online.m)", ME.message);
            end
            drawnow;
        end

        function recordReference(app, ~, ~)
            if app.RecordingActive
                app.StatusLabel.Text = "Recording active. Press Stop when finished speaking.";
                return;
            end
            if exist('audiorecorder', 'file') ~= 2 && exist('audiorecorder', 'builtin') ~= 5
                app.StatusLabel.Text = "Microphone unavailable in this MATLAB environment. Please use 'Upload WAV'.";
                return;
            end
            try
                app.Recorder = audiorecorder(app.Config.fs, 16, 1);
                record(app.Recorder);
                app.RecordingActive = true;
                app.StatusLabel.Text = "Recording audio... speak clearly for 3–5 seconds, then press Stop.";
            catch ME
                app.StatusLabel.Text = sprintf("Microphone access failed: %s. Use 'Upload WAV' instead.", ME.message);
            end
            drawnow;
        end

        function stopRecording(app, ~, ~)
            if isempty(app.Recorder) || ~isvalid(app.Recorder)
                app.StatusLabel.Text = "No active recording to stop.";
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
                    app.StatusLabel.Text = "Recording produced no samples. Check microphone.";
                    return;
                end
                fs = app.Config.fs;
                if sqrt(mean(audio.^2)) < 1e-4
                    app.StatusLabel.Text = "Recorded audio is silent. Check input device.";
                    return;
                end
                dur = numel(audio) / fs;
                if dur < app.Config.spk_min_dur_s
                    app.StatusLabel.Text = sprintf("Recording too short (%.1f s); minimum required is %.1f s.", dur, app.Config.spk_min_dur_s);
                    return;
                end
                app.setReferenceAudio(audio, fs, "Microphone Recording");
                app.StatusLabel.Text = sprintf("Reference recorded (%.1f s). Ready to Generate.", dur);
            catch ME
                app.StatusLabel.Text = sprintf("Failed to finalize recording: %s", ME.message);
            end
            drawnow;
        end

        function loadReference(app, ~, ~)
            [file, path] = uigetfile({'*.wav;*.flac;*.mp3;*.m4a;*.ogg', 'Audio Files (*.wav, *.flac, *.mp3)'}, 'Select Reference Voice File');
            if isequal(file, 0)
                return;
            end
            fullPath = fullfile(path, file);
            try
                [x, fs] = audioread(fullPath);
            catch ME
                app.StatusLabel.Text = sprintf("Failed to read '%s': %s", file, ME.message);
                return;
            end
            try
                [x_clean, fs_clean] = preprocess_signal(x, fs, app.Config.fs);
                dur = numel(x_clean) / fs_clean;
                if dur < app.Config.spk_min_dur_s
                    app.StatusLabel.Text = sprintf("Audio too short (%.2f s); minimum is %.1f s.", dur, app.Config.spk_min_dur_s);
                    return;
                end
                app.setReferenceAudio(x_clean, fs_clean, string(file));
                app.StatusLabel.Text = sprintf("Reference loaded: %s (%.1f s). Ready to Generate.", file, dur);
            catch ME
                app.StatusLabel.Text = sprintf("Reference preprocessing failed: %s", ME.message);
            end
            drawnow;
        end

        function setReferenceAudio(app, audio, fs, sourceName)
            newFingerprint = app.computeFingerprint(audio, fs);
            if ~strcmp(app.ReferenceFingerprint, newFingerprint)
                % Invalidate cached speaker embedding on reference change
                app.CachedSpeakerEmbedding = [];
                app.ReferenceFingerprint = newFingerprint;
                fprintf("[VoiceClonerApp] Reference audio updated — speaker embedding cache invalidated.\n");
            end
            app.ReferenceAudio = double(audio(:));
            app.ReferenceFs = fs;
            app.ReferenceSource = sourceName;
            app.updateReferencePlot();
        end

        function fp = computeFingerprint(~, audio, fs)
            % Deterministic fingerprint derived from length, RMS, and distributed samples
            N = numel(audio);
            rmsVal = sqrt(mean(audio.^2));
            idx = round(linspace(1, N, min(N, 50)));
            sampleSum = sum(abs(audio(idx)));
            fp = sprintf("len_%d_fs_%d_rms_%.5f_sum_%.5f", N, fs, rmsVal, sampleSum);
        end

        function playReference(app, ~, ~)
            if isempty(app.ReferenceAudio)
                app.StatusLabel.Text = "No reference audio loaded.";
                return;
            end
            try
                sound(app.ReferenceAudio, app.ReferenceFs);
                app.StatusLabel.Text = "Playing reference audio...";
            catch ME
                app.StatusLabel.Text = sprintf("Playback failed: %s", ME.message);
            end
        end

        function generateSpeech(app, ~, ~)
            if isempty(app.ReferenceAudio)
                app.StatusLabel.Text = "Load or record reference speech first.";
                return;
            end
            textVal = strtrim(strjoin(string(app.TargetTextArea.Value), " "));
            if strlength(textVal) == 0
                app.StatusLabel.Text = "Target text is empty.";
                return;
            end

            hasCachedEmb = ~isempty(app.CachedSpeakerEmbedding);

            try
                tStart = tic;
                if hasCachedEmb
                    app.StatusLabel.Text = "Reusing cached speaker embedding — synthesizing voice...";
                    drawnow;
                    result = generate_voice_from_embedding(app.CachedSpeakerEmbedding, textVal, app.LoadedModels, app.Config);
                else
                    app.StatusLabel.Text = "Extracting speaker embedding and synthesizing voice...";
                    drawnow;
                    result = generate_voice(app.ReferenceAudio, app.ReferenceFs, textVal, app.LoadedModels, app.Config);
                    % Cache embedding for subsequent sentences
                    app.CachedSpeakerEmbedding = result.speakerEmbedding;
                end

                app.CurrentResult = result;
                app.GeneratedAudio = result.waveform;
                app.GeneratedFs = result.sampleRate;
                app.GenerationMetrics = result.metrics;

                app.updateOutputPlot();
                app.updateMelPlot(result.acousticFeatures);

                elapsed = toc(tStart);
                app.StatusLabel.Text = sprintf("Generated %.2f s audio in %.2f s (RTF: %.2fx)", ...
                    result.metadata.audioDurationSeconds, result.metrics.totalTime_s, result.metadata.realTimeFactor);
                app.MetricsLabel.Text = sprintf("pre=%.2fs | spk=%.2fs | tok=%.2fs | enc=%.2fs | dec=%.2fs | voc=%.2fs (wall=%.2fs)", ...
                    result.metrics.preprocessingTime_s, result.metrics.speakerEncoderTime_s, ...
                    result.metrics.tokenizationTime_s, result.metrics.encoderTime_s, ...
                    result.metrics.decoderTime_s, result.metrics.vocoderTime_s, elapsed);

            catch ME
                app.StatusLabel.Text = sprintf("Generation error: %s", ME.message);
                app.MetricsLabel.Text = "Synthesis aborted.";
                disp(getReport(ME, 'extended'));
            end
            drawnow;
        end

        function playOutput(app, ~, ~)
            if isempty(app.GeneratedAudio)
                app.StatusLabel.Text = "No synthesized audio to play.";
                return;
            end
            try
                sound(app.GeneratedAudio, app.GeneratedFs);
                app.StatusLabel.Text = "Playing synthesized speech output...";
            catch ME
                app.StatusLabel.Text = sprintf("Playback failed: %s", ME.message);
            end
        end

        function saveOutput(app, ~, ~)
            if isempty(app.GeneratedAudio)
                app.StatusLabel.Text = "No generated audio available to save.";
                return;
            end
            [file, path] = uiputfile('*.wav', 'Save Synthesized Speech As');
            if isequal(file, 0)
                return;
            end
            fullDest = fullfile(path, file);
            try
                audiowrite(fullDest, app.GeneratedAudio, app.GeneratedFs);
                app.StatusLabel.Text = sprintf("Saved to: %s", fullDest);
            catch ME
                app.StatusLabel.Text = sprintf("Save failed: %s", ME.message);
            end
        end

        function updateReferencePlot(app)
            if isempty(app.ReferenceAxes) || ~isvalid(app.ReferenceAxes)
                return;
            end
            if isempty(app.ReferenceAudio)
                cla(app.ReferenceAxes);
                return;
            end
            t = (0:numel(app.ReferenceAudio)-1) / app.ReferenceFs;
            plot(app.ReferenceAxes, t, app.ReferenceAudio, 'Color', [0.15 0.35 0.75]);
            xlabel(app.ReferenceAxes, 'Time (seconds)');
            ylabel(app.ReferenceAxes, 'Amplitude');
            title(app.ReferenceAxes, sprintf('Reference Speech (%s, %.1f s)', app.ReferenceSource, numel(app.ReferenceAudio)/app.ReferenceFs));
            grid(app.ReferenceAxes, 'on');
        end

        function updateOutputPlot(app)
            if isempty(app.OutputAxes) || ~isvalid(app.OutputAxes)
                return;
            end
            if isempty(app.GeneratedAudio)
                cla(app.OutputAxes);
                return;
            end
            t = (0:numel(app.GeneratedAudio)-1) / app.GeneratedFs;
            plot(app.OutputAxes, t, app.GeneratedAudio, 'Color', [0.1 0.65 0.3]);
            xlabel(app.OutputAxes, 'Time (seconds)');
            ylabel(app.OutputAxes, 'Amplitude');
            title(app.OutputAxes, sprintf('Synthesized Waveform (%.2f s, %d Hz)', ...
                numel(app.GeneratedAudio)/app.GeneratedFs, app.GeneratedFs));
            grid(app.OutputAxes, 'on');
        end

        function updateMelPlot(app, mel)
            if isempty(app.MelAxes) || ~isvalid(app.MelAxes)
                return;
            end
            try
                cla(app.MelAxes);
                imagesc(app.MelAxes, mel);
                axis(app.MelAxes, 'tight');
                xlabel(app.MelAxes, 'Temporal Frames');
                ylabel(app.MelAxes, 'Mel Frequency Bins (80-7600 Hz)');
                title(app.MelAxes, sprintf('Synthesized Mel Features [80 x %d]', size(mel, 2)));
                colormap(app.MelAxes, 'parula');
                colorbar(app.MelAxes);
            catch
            end
        end
    end
end
