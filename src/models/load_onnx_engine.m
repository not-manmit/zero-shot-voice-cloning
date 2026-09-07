function net = load_onnx_engine(model_path)
%LOAD_ONNX_ENGINE Import a zero-shot TTS ONNX network.
%   net = LOAD_ONNX_ENGINE(model_path) imports the model without generating
%   code. The model file is expected to be downloaded by scripts/download_weights.m.

arguments
    model_path (1,1) string = "models/weights/tts_model.onnx"
end

if ~isfile(model_path)
    error("load_onnx_engine:MissingModel", ...
        "ONNX model not found: %s. Run scripts/download_weights.m in MATLAB Online.", model_path);
end

net = importONNXNetwork(model_path, OutputLayerType="regression");
end
