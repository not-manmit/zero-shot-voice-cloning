function [net, meta] = import_onnx_model(modelPath, options)
%IMPORT_ONNX_MODEL Centralized, version-aware ONNX importer for MATLAB.
%
%   [net, meta] = IMPORT_ONNX_MODEL(modelPath)
%   [net, meta] = IMPORT_ONNX_MODEL(modelPath, options)
%
%   Design Principles:
%     1. Prefers current MATLAB API: importNetworkFromONNX (R2023b+ recommended)
%     2. Falls back to importONNXNetwork only if the modern API is absent
%     3. Never silently hides a failed modern import behind an obsolete fallback
%     4. Returns rich diagnostic metadata (InputNames, OutputNames, shapes, API used)
%     5. Fails with actionable diagnostic error messages
%
%   Outputs:
%     net  – imported dlnetwork or dagnet object
%     meta – struct with forensic inspection details:
%            .path, .fileSizeBytes, .apiUsed, .networkClass,
%            .inputNames, .outputNames, .isInitialized, .error

arguments
    modelPath (1,1) string
    options.TargetNetwork string = "dlnetwork"
    options.OutputLayerType string = "regression"
end

meta = struct();
meta.path = modelPath;
meta.fileSizeBytes = 0;
meta.apiUsed = "none";
meta.networkClass = "unknown";
meta.inputNames = strings(0,1);
meta.outputNames = strings(0,1);
meta.isInitialized = false;
meta.error = "";

% 1. File existence check
if ~isfile(modelPath)
    errId = "import_onnx_model:FileNotFound";
    errMsg = sprintf("ONNX model file not found:\n  Path: %s\nRun scripts/setup_matlab_online.m or scripts/download_weights.m to acquire it.", modelPath);
    meta.error = errMsg;
    error(errId, "%s", errMsg);
end

fileInfo = dir(modelPath);
meta.fileSizeBytes = fileInfo.bytes;
if meta.fileSizeBytes < 1024
    errId = "import_onnx_model:FileTooSmall";
    errMsg = sprintf("ONNX file is suspiciously small (%d bytes):\n  Path: %s\nThe file may be a broken download or placeholder.", meta.fileSizeBytes, modelPath);
    meta.error = errMsg;
    error(errId, "%s", errMsg);
end

% 2. Detect available import functions
hasModernImport = exist('importNetworkFromONNX', 'file') == 2 || exist('importNetworkFromONNX', 'builtin') == 5;
hasLegacyImport = exist('importONNXNetwork', 'file') == 2 || exist('importONNXNetwork', 'builtin') == 5;

if ~hasModernImport && ~hasLegacyImport
    errId = "import_onnx_model:NoImporterAvailable";
    errMsg = sprintf(["No ONNX import function found in this MATLAB session.\n" ...
        "Model: %s\n" ...
        "Required: Deep Learning Toolbox Converter for ONNX Model Format\n" ...
        "Install via: MATLAB Home > Add-Ons > Get Add-Ons > search 'ONNX'."], modelPath);
    meta.error = errMsg;
    error(errId, "%s", errMsg);
end

net = [];
importSuccess = false;
lastError = [];

% 3. Attempt importNetworkFromONNX (Primary, modern API)
if hasModernImport
    try
        meta.apiUsed = "importNetworkFromONNX";
        % importNetworkFromONNX creates dlnetwork directly in current releases
        try
            net = importNetworkFromONNX(modelPath, TargetNetwork=options.TargetNetwork, OutputLayerType=options.OutputLayerType);
        catch
            % Some releases do not accept OutputLayerType for importNetworkFromONNX
            net = importNetworkFromONNX(modelPath, TargetNetwork=options.TargetNetwork);
        end
        importSuccess = true;
    catch ME
        lastError = ME;
        meta.error = sprintf("importNetworkFromONNX failed: %s", ME.message);
    end
end

% 4. Fallback to importONNXNetwork only if modern import was absent or genuinely failed
if ~importSuccess && hasLegacyImport
    try
        meta.apiUsed = "importONNXNetwork (fallback)";
        try
            net = importONNXNetwork(modelPath, OutputLayerType=options.OutputLayerType, TargetNetwork=options.TargetNetwork);
        catch
            net = importONNXNetwork(modelPath, OutputLayerType=options.OutputLayerType);
        end
        importSuccess = true;
    catch ME_legacy
        if isempty(lastError)
            lastError = ME_legacy;
        else
            lastError = MException("import_onnx_model:BothImportersFailed", ...
                sprintf("Both modern and legacy importers failed for %s.\nModern error: %s\nLegacy error: %s", ...
                modelPath, lastError.message, ME_legacy.message));
        end
        meta.error = lastError.message;
    end
end

if ~importSuccess
    errId = "import_onnx_model:ImportFailed";
    errMsg = sprintf(["Failed to import ONNX model:\n" ...
        "  File: %s (%.1f MB)\n" ...
        "  API attempted: %s\n" ...
        "  MATLAB release: %s\n" ...
        "  Error details: %s\n" ...
        "  Likely causes:\n" ...
        "    - Missing 'Deep Learning Toolbox Converter for ONNX Model Format'\n" ...
        "    - Unsupported ONNX opset (SpeechT5 fp32 uses opset 14, quantized may use opset 17+)\n" ...
        "    - Corrupted download file (re-download with scripts/download_weights(true))"], ...
        modelPath, meta.fileSizeBytes / 1e6, meta.apiUsed, version, lastError.message);
    error(errId, "%s", errMsg);
end

% 5. Inspect imported network properties
meta.networkClass = class(net);
if isprop(net, 'InputNames')
    meta.inputNames = string(net.InputNames);
end
if isprop(net, 'OutputNames')
    meta.outputNames = string(net.OutputNames);
end
if isprop(net, 'Initialized')
    meta.isInitialized = net.Initialized;
else
    meta.isInitialized = true; % Older DAG objects consider themselves initialized
end

end
