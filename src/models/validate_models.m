function report = validate_models()
%VALIDATE_MODELS Load and inspect the ONNX model files required by the pipeline.
%
% Report structure includes inputs/outputs and a boolean ok flag.

cfg = pipeline_config();
report = struct('ok', false, 'models', struct());
modelNames = {'encoder', 'decoder', 'decoder_kv', 'vocoder', 'spk_encoder'};
paths = {cfg.paths.encoder, cfg.paths.decoder, cfg.paths.decoder_kv, cfg.paths.vocoder, cfg.paths.spk_encoder};

for i = 1:numel(modelNames)
    name = modelNames{i};
    path = paths{i};
    entry = struct('path', path, 'exists', false, 'ok', false, 'inputs', {}, 'outputs', {}, 'error', '');

    if ~isfile(path)
        entry.error = sprintf('Model file missing: %s', path);
        report.models.(name) = entry;
        continue;
    end

    entry.exists = true;
    try
        net = importONNXNetwork(path, OutputLayerType="regression");
        entry.ok = true;
        entry.inputs = describe_network_io(net, 'input');
        entry.outputs = describe_network_io(net, 'output');
        report.models.(name) = entry;
    catch ME
        entry.error = sprintf('ONNX import failed for %s: %s', path, ME.message);
        report.models.(name) = entry;
    end
end

allGood = true;
fn = fieldnames(report.models);
for i = 1:numel(fn)
    if ~report.models.(fn{i}).ok && ~isempty(report.models.(fn{i}).error)
        allGood = false;
    end
end
report.ok = allGood;
end

function details = describe_network_io(net, kind)
if strcmp(kind, 'input')
    names = net.InputNames;
    if isempty(names)
        details = {'<unnamed>'};
        return;
    end
    details = cellstr(names);
    return;
end

names = net.OutputNames;
if isempty(names)
    details = {'<unnamed>'};
    return;
end
details = cellstr(names);
end
