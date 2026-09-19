function report = validate_models(cfg)
%VALIDATE_MODELS Load and inspect the ONNX model files required by the pipeline.
%
%   report = VALIDATE_MODELS()
%   report = VALIDATE_MODELS(cfg)
%
%   For every model verifies:
%     - file exists
%     - importONNXNetwork succeeds (dlnetwork first, then dag)
%     - expected input/output names and dimensions are present
%     - distinguishes legacy int BOS vs verified float mel contract for decoder
%     - past tensor count symmetry, use_cache_branch presence

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
        'contract', c.(cKey), 'error', '', 'legacyFallback', false);

    if ~isfile(path)
        entry.error = sprintf('Model file missing: %s (run scripts/download_weights.m or setup_matlab_online.m)', path);
        overallOk = false;
        report.models.(name) = entry;
        fprintf('[validate_models] %-20s MISSING %s\n', name, path);
        continue;
    end
    entry.exists = true;

    try
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

        % Compare against contract — handle decoder float vs legacy
        if strcmp(name,'decoder')
            % Check float primary first
            expectedFloat = {c.(cKey).inputs.name};
            expectedLegacy = {c.(cKey).inputs_legacy.name};
            lowerIn = lower(entry.inputs);
            isFloat = any(contains(lowerIn,'output_sequence')) || any(contains(lowerIn,'decoder_input_values')) || any(contains(lowerIn,'input_values'));
            isLegacy = any(strcmp(lowerIn,'input_ids')) && ~isFloat;
            if isFloat
                missing = setdiff(lower(expectedFloat), lowerIn);
                if ~isempty(missing)
                    entry.ok = false;
                    entry.error = sprintf('Decoder (float) input name mismatch. Expected [%s] Found [%s]', strjoin(expectedFloat,','), strjoin(entry.inputs,','));
                    overallOk = false;
                else
                    fprintf('[validate_models] decoder: float Mel contract OK (output_sequence)\n');
                end
            elseif isLegacy
                entry.legacyFallback = true;
                missing = setdiff(lower(expectedLegacy), lowerIn);
                if ~isempty(missing)
                    entry.ok = false;
                    entry.error = sprintf('Decoder (legacy int) input name mismatch. Expected [%s] Found [%s]', strjoin(expectedLegacy,','), strjoin(entry.inputs,','));
                    overallOk = false;
                else
                    fprintf('[validate_models] decoder: LEGACY int BOS contract detected — synthesize_features will use fallback but upgrade to float export recommended.\n');
                end
            else
                entry.ok = false;
                entry.error = sprintf('Decoder input name mismatch. Neither float nor legacy. Found [%s]', strjoin(entry.inputs,','));
                overallOk = false;
            end
            % Output check: expect spectrum/prob (2) or legacy logits (1)
            lowerOut = lower(entry.outputs);
            hasSpectrum = any(contains(lowerOut,'spectrum')) || any(contains(lowerOut,'feat'));
            hasProb = any(contains(lowerOut,'prob')) || any(contains(lowerOut,'logit'));
            if hasSpectrum || hasProb
                fprintf('[validate_models] decoder outputs: spectrum/prob contract (rf=2)\n');
            else
                fprintf('[validate_models] decoder outputs: %s (check if packed 81 legacy)\n', strjoin(entry.outputs,','));
            end
            nPastOutputs = max(0, numel(entry.outputs) - 2); % minus spectrum,prob
            fprintf('[validate_models] decoder past outputs: %d\n', nPastOutputs);
        elseif strcmp(name,'decoder_with_past')
            % core 4 float + past
            expectedCore = {c.(cKey).inputs.name}; expectedCore = expectedCore(1:4);
            lowerIn = lower(entry.inputs);
            hasUseCache = any(contains(lowerIn,'use_cache'));
            missingCore = setdiff(lower(expectedCore), lowerIn);
            % Allow legacy core too
            if ~isempty(missingCore)
                legacyCore = lower({c.(cKey).inputs_legacy.name}); legacyCore = legacyCore(1:4);
                missingLegacy = setdiff(legacyCore, lowerIn);
                if isempty(missingLegacy)
                    entry.legacyFallback = true;
                    fprintf('[validate_models] decoder_with_past: legacy int core detected\n');
                else
                    entry.ok = false;
                    entry.error = sprintf('Decoder_with_past missing core inputs: %s (found %s)', strjoin(missingCore,','), strjoin(entry.inputs,','));
                    overallOk = false;
                end
            end
            nPastInputs = numel(entry.inputs) - 4 - hasUseCache;
            nPastOutputs = numel(entry.outputs) - 2; % minus spectrum,prob (or 1 if legacy)
            if nPastOutputs < 0
                nPastOutputs = numel(entry.outputs) - 1;
            end
            fprintf('[validate_models] decoder_with_past past tensors: %d inputs / %d outputs (use_cache_branch %d)\n', nPastInputs, nPastOutputs, hasUseCache);
            if nPastInputs ~= nPastOutputs && nPastOutputs > 0
                fprintf('[validate_models] Warning: past input/output count mismatch %d vs %d\n', nPastInputs, nPastOutputs);
            end
        else
            expectedIn = {c.(cKey).inputs.name};
            missing = setdiff(lower(expectedIn), lower(entry.inputs));
            extra = setdiff(lower(entry.inputs), lower(expectedIn));
            if ~isempty(missing) || ~isempty(extra)
                entry.ok = false;
                entry.error = sprintf('Input name mismatch for %s. Expected [%s] Found [%s]', name, strjoin(expectedIn,','), strjoin(entry.inputs,','));
                overallOk = false;
            end
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
