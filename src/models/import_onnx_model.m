function [net, meta] = import_onnx_model(modelPath, options)
%IMPORT_ONNX_MODEL Centralized, version-aware ONNX importer for MATLAB R2026a+.
%
%   [net, meta] = IMPORT_ONNX_MODEL(modelPath)
%   [net, meta] = IMPORT_ONNX_MODEL(modelPath, options)
%
%   Design Principles:
%     1. Prefers current MATLAB API: importNetworkFromONNX (R2026a native)
%     2. Avoids calling legacy importONNXNetwork when modern importer is present,
%        preventing legacy "Subscripted assignment between dissimilar structures" errors.
%     3. Does not classify a partially imported uninitialized or placeholder-containing
%        network as a fully inference-ready import.
%     4. Returns rich diagnostic metadata (InputNames, OutputNames, shapes, API used)
%     5. Fails with actionable diagnostic error messages.
%
%   Outputs:
%     net  – imported dlnetwork object
%     meta – struct with forensic inspection details:
%            .path, .fileSizeBytes, .apiUsed, .networkClass,
%            .inputNames, .outputNames, .isInitialized, .hasPlaceholderLayers, .error

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
meta.hasPlaceholderLayers = false;
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

% 3. Modern import via importNetworkFromONNX (Primary, preferred for R2026a)
if hasModernImport
    meta.apiUsed = "importNetworkFromONNX";
    try
        % In MATLAB R2026a, importNetworkFromONNX accepts model path directly
        net = importNetworkFromONNX(modelPath);
        importSuccess = true;
    catch ME
        % If direct call failed, try with TargetNetwork if supported
        try
            net = importNetworkFromONNX(modelPath, TargetNetwork="dlnetwork");
            importSuccess = true;
        catch
            lastError = ME;
            meta.error = sprintf("importNetworkFromONNX failed: %s", ME.message);
        end
    end
end

% 4. Fallback to legacy importONNXNetwork ONLY on ancient MATLAB where modern is completely absent
if ~importSuccess && ~hasModernImport && hasLegacyImport
    meta.apiUsed = "importONNXNetwork (legacy fallback)";
    try
        net = importONNXNetwork(modelPath, OutputLayerType=options.OutputLayerType);
        importSuccess = true;
    catch ME_legacy
        lastError = ME_legacy;
        meta.error = sprintf("importONNXNetwork failed: %s", ME_legacy.message);
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
        "    - Unsupported ONNX operator or opset\n" ...
        "    - Corrupted download file (re-download with scripts/download_weights(true))"], ...
        modelPath, meta.fileSizeBytes / 1e6, meta.apiUsed, version, lastError.message);
    error(errId, "%s", errMsg);
end

% 5. Inspect imported network properties and initialization state
meta.networkClass = class(net);
if isprop(net, 'InputNames')
    meta.inputNames = string(net.InputNames);
end
if isprop(net, 'OutputNames')
    meta.outputNames = string(net.OutputNames);
end

isInit = true;
if isprop(net, 'Initialized')
    try
        isInit = logical(net.Initialized);
    catch
        isInit = false;
    end
end
if isempty(isInit) || ~isscalar(isInit)
    isInit = isscalar(isInit) && isInit(1);
end

hasPlaceholders = false;
if isprop(net, 'Layers')
    try
        layers = net.Layers;
        if ~isempty(layers)
            layerClasses = string(arrayfun(@class, layers, 'UniformOutput', false));
            layerNames = string({layers.Name});
            isPhClass = contains(layerClasses, 'placeholder', 'IgnoreCase', true);
            isPhName = contains(layerNames, 'placeholder', 'IgnoreCase', true);
            hasPlaceholders = any(isPhClass) || any(isPhName);
        end
    catch
    end
end

hasPlaceholders = isscalar(hasPlaceholders) && logical(hasPlaceholders);
meta.hasPlaceholderLayers = hasPlaceholders;

% Do not classify a partially imported uninitialized or placeholder-bearing network
% as fully inference-ready
meta.isInitialized = isscalar(isInit) && logical(isInit) && ~hasPlaceholders;

end
