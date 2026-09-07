function save_path = download_weights(model_url, model_filename)
%DOWNLOAD_WEIGHTS Download an ONNX model into MATLAB Drive.
%   Run this utility in MATLAB Online. Model files are intentionally ignored
%   by Git because they can exceed repository and MATLAB Drive limits.

arguments
    model_url (1,1) string
    model_filename (1,1) string = "tts_model.onnx"
end

weights_dir = fullfile(pwd, "models", "weights");
if ~isfolder(weights_dir)
    mkdir(weights_dir);
end
save_path = fullfile(weights_dir, model_filename);
if isfile(save_path)
    fprintf("Weights already exist: %s\n", save_path);
    return;
end

fprintf("Downloading model weights to %s\n", save_path);
websave(save_path, model_url);
end
