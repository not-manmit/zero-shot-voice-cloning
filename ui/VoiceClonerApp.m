classdef VoiceClonerApp < handle
    %VOICECLONERAPP Minimal App Designer-compatible MATLAB UI entry point.

    properties
        UIFigure matlab.ui.Figure
        TextArea matlab.ui.control.TextArea
        StatusLabel matlab.ui.control.Label
        ReferenceAxes matlab.ui.control.UIAxes
        SpectrogramAxes matlab.ui.control.UIAxes
        ReferenceAudio double = []
        ReferenceFs double = 24000
    end

    methods
        function app = VoiceClonerApp
            app.UIFigure = uifigure(Name="Zero-Shot Voice Cloner", Position=[100 100 980 620]);
            app.TextArea = uitextarea(app.UIFigure, Position=[30 540 430 45], Placeholder="Target text");
            uibutton(app.UIFigure, Text="Load reference", Position=[30 495 140 30], ...
                ButtonPushedFcn=@app.loadReference);
            uibutton(app.UIFigure, Text="Synthesize", Position=[185 495 110 30], ...
                ButtonPushedFcn=@app.synthesize);
            app.StatusLabel = uilabel(app.UIFigure, Text="Ready", Position=[310 500 300 22]);
            app.ReferenceAxes = uiaxes(app.UIFigure, Position=[30 245 430 220], Title="Reference waveform");
            app.SpectrogramAxes = uiaxes(app.UIFigure, Position=[500 245 430 220], Title="Mel spectrogram");
        end

        function loadReference(app, ~, ~)
            [file, path] = uigetfile("*.wav", "Select reference speech");
            if isequal(file, 0)
                return;
            end
            [x, fs] = audioread(fullfile(path, file));
            [app.ReferenceAudio, app.ReferenceFs] = preprocess_signal(x, fs);
            t = (0:numel(app.ReferenceAudio)-1) / app.ReferenceFs;
            plot(app.ReferenceAxes, t, app.ReferenceAudio);
            app.StatusLabel.Text = "Reference loaded";
            drawnow;
        end

        function synthesize(app, ~, ~)
            if isempty(app.ReferenceAudio)
                app.StatusLabel.Text = "Load a reference WAV first";
                return;
            end
            target_text = string(strjoin(app.TextArea.Value, " "));
            if strlength(strtrim(target_text)) == 0
                app.StatusLabel.Text = "Enter target text first";
                return;
            end
            app.StatusLabel.Text = "Extracting features...";
            drawnow;
            [mel_matrix, ~, ~, ~] = stft_analysis(app.ReferenceAudio, app.ReferenceFs);
            imagesc(app.SpectrogramAxes, mel_matrix);
            axis(app.SpectrogramAxes, "xy");
            colorbar(app.SpectrogramAxes);
            app.StatusLabel.Text = "Features ready; configure ONNX model for inference";
            drawnow;
        end
    end
end
