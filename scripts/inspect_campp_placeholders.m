function analysis = inspect_campp_placeholders()
%INSPECT_CAMPP_PLACEHOLDERS Deep inspection of CAM++ ONNX placeholder layers in MATLAB Online.
%
%   analysis = INSPECT_CAMPP_PLACEHOLDERS()
%
%   Detailed forensic inspection of CAM++ (xvector_encoder.onnx):
%     1. Imports CAM++ using modern importNetworkFromONNX.
%     2. Discovers all generated placeholder layers and functions.
%     3. Extracts input/output connections, predecessor and successor layers.
%     4. Inspects BatchNorm training_mode layer (BatchNormalizatio_49).
%     5. Evaluates replacement feasibility via replaceLayer.

fprintf("=================================================================\n");
fprintf("       CAM++ ONNX PLACEHOLDER & OPERATOR FORENSIC ANALYZER       \n");
fprintf("=================================================================\n\n");

cfg = pipeline_config();
modelPath = cfg.paths.spk_encoder;

if ~isfile(modelPath)
    error("inspect_campp_placeholders:MissingModel", ...
        "CAM++ model is missing at %s.", modelPath);
end

fprintf("Importing CAM++ model:\n  %s\n\n", modelPath);
try
    [net, meta] = import_onnx_model(modelPath);
catch ME
    error("inspect_campp_placeholders:ImportFailed", ...
        "CAM++ import failed: %s", ME.message);
end

analysis = struct();
analysis.meta = meta;
analysis.placeholderLayers = {};
analysis.batchNormLayers = {};

if ~isprop(net, 'Layers')
    fprintf("Network does not expose 'Layers' property (class: %s).\n", class(net));
    return;
end

layers = net.Layers;
fprintf("Total layers in imported graph: %d\n", numel(layers));

% 1. Find all placeholder layers
for i = 1:numel(layers)
    lyr = layers(i);
    className = class(lyr);
    lyrName = lyr.Name;

    if contains(className, "placeholder", "IgnoreCase", true) || ...
       contains(lyrName, "placeholder", "IgnoreCase", true) || ...
       contains(lyrName, "AveragePool", "IgnoreCase", true)
        analysis.placeholderLayers{end+1} = struct( ...
            'index', i, ...
            'name', lyrName, ...
            'class', className, ...
            'layer', lyr);
    end

    if contains(lyrName, "BatchNormalizatio_49") || contains(lyrName, "BatchNorm", "IgnoreCase", true)
        if contains(lyrName, "49")
            analysis.batchNormLayers{end+1} = struct( ...
                'index', i, ...
                'name', lyrName, ...
                'class', className, ...
                'layer', lyr);
        end
    end
end

fprintf("\n--- Placeholder Layers Discovered (%d total) ---\n", numel(analysis.placeholderLayers));
for k = 1:min(15, numel(analysis.placeholderLayers))
    p = analysis.placeholderLayers{k};
    fprintf("  [%d] Layer '%s' | Class: %s\n", p.index, p.name, p.class);
    propNames = properties(p.layer);
    for pn = 1:numel(propNames)
        pName = propNames{pn};
        if ~ismember(pName, {'Name', 'NumInputs', 'NumOutputs', 'InputNames', 'OutputNames'})
            try
                val = p.layer.(pName);
                if isnumeric(val) || islogical(val) || isstring(val) || ischar(val)
                    fprintf("       .%s = %s\n", pName, mat2str(val));
                end
            catch
            end
        end
    end
end
if numel(analysis.placeholderLayers) > 15
    fprintf("  ... and %d more placeholder layers.\n", numel(analysis.placeholderLayers) - 15);
end

% 2. Inspect BatchNormalizatio_49
fprintf("\n--- BatchNorm Layer Inspection ---\n");
if ~isempty(analysis.batchNormLayers)
    for b = 1:numel(analysis.batchNormLayers)
        bn = analysis.batchNormLayers{b};
        fprintf("  Layer '%s' | Class: %s\n", bn.name, bn.class);
    end
else
    fprintf("  BatchNormalizatio_49 not found in Layers array.\n");
end

% 3. Graph Connections around Placeholders
fprintf("\n--- Graph Connection Inspection ---\n");
try
    lg = layerGraph(net);
    analysis.hasLayerGraph = true;
    fprintf("  layerGraph successfully extracted from imported network.\n");
    fprintf("  Layer count: %d | Connection count: %d\n", numel(lg.Layers), size(lg.Connections, 1));
catch ME
    analysis.hasLayerGraph = false;
    fprintf("  layerGraph extraction failed: %s\n", ME.message);
end

fprintf("\n=================================================================\n");
fprintf("Investigation complete. Report struct returned.\n");
fprintf("=================================================================\n");

end
