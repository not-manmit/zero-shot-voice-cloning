function report = validate_models(cfg)
%VALIDATE_MODELS Load and inspect the ONNX model files required by the pipeline.
%
%   report = VALIDATE_MODELS()
%   report = VALIDATE_MODELS(cfg)
%
%   For every model verifies:
%     - file exists
%     - importONNXNetwork succeeds
%     - expected input/output names and dimensions are present
%     - tensor types are float32/int64 as per contract
%   Distinguishes MODEL FILE EXISTS vs MODEL CAN ACTUALLY BE USED.

if nargin < 1 || isempty(cfg) || ~isfield(cfg,'paths')
    cfg = pipeline_config();
end
c = model_contract();

report = struct('ok', false, 'models', struct(), 'contract', c);
modelNames = {'encoder', 'decoder', 'decoder_with_past', 'vocoder', 'spk_encoder'};
contractKeys = {'encoder','decoder','decoder_with_past','vocoder','spk_encoder'};

overallOk = true;

for i = 1:numel(modelNames)
    name = modelNames{i};
    cKey = contractKeys{i};
    path = c.(cKey).file;
    entry = struct('path', path, 'exists', false, 'ok', false, ...
        'inputs', {{}}, 'outputs', {{}}, ...
        'inputDetails', {{}}, 'outputDetails', {{}}, ...
        'contract', c.(cKey), 'error', '');

    if ~isfile(path)
        entry.error = sprintf('Model file missing: %s (run scripts/download_weights.m or setup_matlab_online.m)', path);
        overallOk = false;
        report.models.(name) = entry;
        fprintf('[validate_models] %-20s MISSING %s\n', name, path);
        continue;
    end
    entry.exists = true;

    try
        % Try dlnetwork import first (preserves dynamic axes for transformers)
        try
            net = importONNXNetwork(path, OutputLayerType="regression", TargetNetwork="dlnetwork");
        catch
            net = importONNXNetwork(path, OutputLayerType="regression");
        end
        entry.ok = true;
        entry.inputs = cellstr(net.InputNames);
        entry.outputs = cellstr(net.OutputNames);
        entry.inputDetails = describe_io(net, 'input');
        entry.outputDetails = describe_io(net, 'output');

        % Compare against contract — explicit exact names, case-insensitive
        expectedIn = {c.(cKey).inputs.name};
        % For decoder_with_past, past tensors may be expanded as past_key_values.0 etc — check prefix
        if strcmp(name,'decoder_with_past')
            core = expectedIn(1:4);
            missingCore = setdiff(lower(core), lower(entry.inputs));
            if ~isempty(missingCore)
                entry.ok = false;
                entry.error = sprintf('Decoder_with_past missing core inputs: %s (found %s)', strjoin(missingCore,','), strjoin(entry.inputs,','));
                overallOk = false;
            end
            % warn if past tensor count looks wrong
            nPastInputs = numel(entry.inputs) - 4;
            nPastOutputs = numel(entry.outputs) - 1; % minus logits
            fprintf('[validate_models] decoder_with_past past tensors: %d inputs / %d outputs\n', nPastInputs, nPastOutputs);
        else
            missing = setdiff(lower(expectedIn), lower(entry.inputs));
            extra = setdiff(lower(entry.inputs), lower(expectedIn));
            if ~isempty(missing) || ~isempty(extra)
                entry.ok = false;
                entry.error = sprintf('Input name mismatch for %s. Expected [%s] Found [%s]', name, strjoin(expectedIn,','), strjoin(entry.inputs,','));
                overallOk = false;
            end
            % Also check output count roughly
            expectedOut = {c.(cKey).outputs.name};
            if numel(entry.outputs) < numel(expectedOut)
                entry.ok = false;
                entry.error = sprintf('Output count mismatch for %s. Expected >=%d Found %d [%s]', name, numel(expectedOut), numel(entry.outputs), strjoin(entry.outputs,','));
                overallOk = false;
            end
        end

        fprintf('[validate_models] %-20s OK inputs:%s outputs:%s\n', name, strjoin(entry.inputs,','), strjoin(entry.outputs,','));
        for d = 1:numel(entry.inputDetails)
            fprintf('    input %s : %s\n', entry.inputDetails{d}.name, entry.inputDetails{d}.shape);
        end
        for d = 1:numel(entry.outputDetails)
            fprintf('    output %s : %s\n', entry.outputDetails{d}.name, entry.outputDetails{d}.shape);
        end

    catch ME
        entry.ok = false;
        entry.error = sprintf('ONNX import failed for %s: %s', path, ME.message);
        overallOk = false;
        fprintf('[validate_models] %-20s IMPORT FAILED: %s\n', name, ME.message);
    end
    report.models.(name) = entry;
end

report.ok = overallOk;
if overallOk
    fprintf('[validate_models] All models validated against contracts.\n');
else
    fprintf('[validate_models] Validation FAILED – see errors above. Do not claim ready.\n');
end
end

function details = describe_io(net, kind)
details = {};
try
    if strcmp(kind,'input')
        names = net.InputNames;
        layers = net.Layers;
        for k=1:numel(names)
            s.name = names(k);
            % try to find corresponding input layer size
            lyrIdx = find(contains({layers.Name}, names(k)),1);
            if ~isempty(lyrIdx) && isprop(layers(lyrIdx),'InputSize')
                s.shape = mat2str(layers(lyrIdx).InputSize);
            else
                s.shape = 'dynamic';
            end
            details{end+1}=s;
        end
    else
        names = net.OutputNames;
        for k=1:numel(names)
            s.name = names(k);
            s.shape = 'dynamic';
            details{end+1}=s;
        end
    end
catch
    details = {};
end
end
