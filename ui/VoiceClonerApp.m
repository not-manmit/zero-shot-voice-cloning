classdef VoiceClonerApp < handle
    %VOICECLONERAPP MATLAB Online UI for zero-shot voice cloning.

    properties
        UIFigure matlab.ui.Figure
        TargetTextArea matlab.ui.control.TextArea
        StatusLabel matlab.ui.control.Label
        ReferenceAxes matlab.ui.control.UIAxes
        OutputAxes matlab.ui.control.UIAxes
        ReferenceAudio double = []
        ReferenceFs double = 16000
        GeneratedAudio double = []
        GeneratedFs double = 16000
        LoadedModels struct = struct()
        Config struct = struct()
        SpeakerEmbedding double = []
        CurrentResult struct = struct()
        Recorder matlab.io.AudioRecorder
        RecordingActive logical = false
    end

    methods
        function app = VoiceClonerApp
            app.Config = pipeline_config();
            app.UIFigure = uifigure(Name="Zero-Shot Voice Cloner", Position=[100 100 980 620]);
            uilabel(app.UIFigure, Text="Reference voice", Position=[30 560 150 22]);
            uibutton(app.UIFigure, Text="Record", Position=[30 520 90 30], ButtonPushedFcn=@app.recordReference);
            uibutton(app.UIFigure, Text="Stop", Position=[130 520 90 30], ButtonPushedFcn=@app.stopRecording);
            uibutton(app.UIFigure, Text="Upload WAV", Position=[230 520 120 30], ButtonPushedFcn=@app.loadReference);
            uibutton(app.UIFigure, Text="Play Reference", Position=[360 520 120 30], ButtonPushedFcn=@app.playReference);

            uilabel(app.UIFigure, Text="Target text", Position=[30 470 150 22]);
            app.TargetTextArea = uitextarea(app.UIFigure, Position=[30 400 430 65], Placeholder="Hello, this is a zero-shot voice cloning demonstration.");
            uibutton(app.UIFigure, Text="Generate", Position=[30 350 120 30], ButtonPushedFcn=@app.generateSpeech);
            uibutton(app.UIFigure, Text="Play Output", Position=[170 350 120 30], ButtonPushedFcn=@app.playOutput);
            uibutton(app.UIFigure, Text="Save WAV", Position=[310 350 120 30], ButtonPushedFcn=@app.saveOutput);
            app.StatusLabel = uilabel(app.UIFigure, Text="Ready", Position=[500 350 400 22]);

            app.ReferenceAxes = uiaxes(app.UIFigure, Position=[30 180 430 150], Title="Reference waveform");
            app.OutputAxes = uiaxes(app.UIFigure, Position=[500 180 430 150], Title="Generated waveform");
            app.loadModels();
        end

        function loadModels(app)
            app.StatusLabel.Text = "Loading models...";
            drawnow;
            try
                app.LoadedModels = load_onnx_engine(app.Config);
                app.StatusLabel.Text = "Models ready";
            catch ME
                app.StatusLabel.Text = sprintf("Model loading failed: %s", ME.message);
            end
            drawnow;
        end

        function recordReference(app, ~, ~)
            if ~isempty(app.Recorder) && isvalid(app.Recorder) && app.RecordingActive
                app.StatusLabel.Text = "Recording already active";
                return;
            end
            try
                app.Recorder = audiorecorder(app.Config.fs, 16, 1);
                record(app.Recorder);
                app.RecordingActive = true;
                app.StatusLabel.Text = "Recording reference...";
            catch ME
                app.StatusLabel.Text = sprintf("Microphone unavailable: %s", ME.message);
            end
            drawnow;
        end

        function stopRecording(app, ~, ~)
            if isempty(app.Recorder) || ~isvalid(app.Recorder)
                app.StatusLabel.Text = "No active recording";
                return;
            end
            stop(app.Recorder);
            app.RecordingActive = false;
            [audio, fs] = getaudiodata(app.Recorder, 'double');
            app.ReferenceAudio = audio(:);
            app.ReferenceFs = fs;
            app.updateReferencePlot();
            app.StatusLabel.Text = "Reference recorded";
            drawnow;
        end

        function loadReference(app, ~, ~)
            [file, path] = uigetfile('*.wav', 'Select reference speech');
            if isequal(file, 0)
                return;
            end
            [x, fs] = audioread(fullfile(path, file));
            [app.ReferenceAudio, app.ReferenceFs] = preprocess_signal(x, fs, app.Config.fs);
            app.updateReferencePlot();
            app.StatusLabel.Text = sprintf('Reference loaded: %s', file);
            drawnow;
        end

        function playReference(app, ~, ~)
            if isempty(app.ReferenceAudio)
                app.StatusLabel.Text = 'No reference audio loaded';
                return;
            end
            sound(app.ReferenceAudio, app.ReferenceFs);
        end

        function generateSpeech(app, ~, ~)
            if isempty(app.ReferenceAudio)
                app.StatusLabel.Text = 'Load or record a reference first';
                return;
            end
            textValue = app.TargetTextArea.Value;
            textValue = strjoin(textValue, " ");
            textValue = string(textValue);
            if strlength(strtrim(textValue)) == 0
                app.StatusLabel.Text = 'Enter target text';
                return;
            end
            app.StatusLabel.Text = 'Generating...';
            drawnow;
            try
                tStart = tic;
                result = generate_voice(app.ReferenceAudio, app.ReferenceFs, textValue, app.LoadedModels, app.Config);
                app.CurrentResult = result;
                app.GeneratedAudio = result.waveform;
                app.GeneratedFs = result.sampleRate;
                app.SpeakerEmbedding = result.speakerEmbedding;
                app.updateOutputPlot();
                app.StatusLabel.Text = sprintf('Generation complete in %.2f s', toc(tStart));
            catch ME
                app.StatusLabel.Text = sprintf('Generation failed: %s', ME.message);
            end
            drawnow;
        end

        function playOutput(app, ~, ~)
            if isempty(app.GeneratedAudio)
                app.StatusLabel.Text = 'Generate speech before playback';
                return;
            end
            sound(app.GeneratedAudio, app.GeneratedFs);
        end

        function saveOutput(app, ~, ~)
            if isempty(app.GeneratedAudio)
                app.StatusLabel.Text = 'No generated output to save';
                return;
            end
            [file, path] = uiputfile('*.wav', 'Save generated speech as');
            if isequal(file, 0)
                return;
            end
            audiowrite(fullfile(path, file), app.GeneratedAudio, app.GeneratedFs);
            app.StatusLabel.Text = sprintf('Saved output to %s', fullfile(path, file));
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
            title(app.ReferenceAxes, 'Reference waveform');
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
            title(app.OutputAxes, 'Generated waveform');
        end
    end
end
