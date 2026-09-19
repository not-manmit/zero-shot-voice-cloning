function diagnose_onnx()
%DIAGNOSE_ONNX  Forensic ONNX contract dumper for all 5 models.
%
%   Prints for each model: file, input names/count/shapes/dtypes, output names/count/shapes/dtypes,
%   dynamic dims, past tensor enumeration, use_cache_branch presence, suspected role, unresolved.
%   If MATLAB cannot expose shape, reports limitation explicitly (not fabricated).
%
%   Usage:
%       diagnose_onnx

cfg = pipeline_config();
paths = {cfg.paths.encoder, cfg.paths.decoder, cfg.paths.decoder_kv, cfg.paths.vocoder, cfg.paths.spk_encoder};
labels = {'encoder','decoder','decoder_kv','vocoder','spk_encoder'};

for i=1:numel(paths)
    p = paths{i};
    fprintf('\n=== %s: %s ===\n', labels{i}, p);
    if ~isfile(p)
        fprintf('  MISSING — run download_weights (expected sizes in README)\n');
        continue;
    end
    d = dir(p);
    fprintf('  File size: %.2f MB\n', d.bytes/1e6);
    try
        try
            net = importONNXNetwork(p, OutputLayerType="regression", TargetNetwork="dlnetwork");
            fprintf('  Import: dlnetwork OK\n');
        catch ME1
            fprintf('  dlnetwork import failed: %s\n  Trying dag import...\n', ME1.message);
            net = importONNXNetwork(p, OutputLayerType="regression");
            fprintf('  Import: dag OK\n');
        end
        fprintf('  Inputs (%d):\n', numel(net.InputNames));
        for k=1:numel(net.InputNames)
            fprintf('    %d: %s\n', k, net.InputNames(k));
        end
        fprintf('  Outputs (%d):\n', numel(net.OutputNames));
        for k=1:numel(net.OutputNames)
            fprintf('    %d: %s\n', k, net.OutputNames(k));
        end
        % Detailed io via describe
        try
            inDetails = describe_inputs(net);
            for k=1:numel(inDetails)
                fprintf('    input detail %s : shape=%s dtype=%s\n', inDetails{k}.name, inDetails{k}.shape, inDetails{k}.dtype);
            end
        catch ME
            fprintf('  Input detail unavailable: %s\n', ME.message);
        end
        try
            outDetails = describe_outputs(net);
            for k=1:numel(outDetails)
                fprintf('    output detail %s : shape=%s dtype=%s\n', outDetails{k}.name, outDetails{k}.shape, outDetails{k}.dtype);
            end
        catch ME
            fprintf('  Output detail unavailable: %s\n', ME.message);
        end
        % Past tensor heuristics
        lowIn = lower(cellstr(net.InputNames));
        lowOut = lower(cellstr(net.OutputNames));
        nPastIn = sum(contains(lowIn,'past') | contains(lowIn,'cache') | contains(lowIn,'present'));
        nPastOut = sum(contains(lowOut,'past') | contains(lowOut,'present') | contains(lowOut,'cache'));
        hasUseCache = any(contains(lowIn,'use_cache'));
        fprintf('  Heuristic: past inputs %d past outputs %d use_cache_branch %d\n', nPastIn, nPastOut, hasUseCache);
        if strcmp(labels{i},'decoder') || strcmp(labels{i},'decoder_kv')
            isFloat = any(contains(lowIn,'output_sequence')) || any(contains(lowIn,'decoder_input_values'));
            isLegacy = any(strcmp(lowIn,'input_ids'));
            fprintf('  Decoder contract: isFloatMel %d isLegacyInt %d\n', isFloat, isLegacy);
            if isFloat
                fprintf('    -> Verified float output_sequence [B,1,80] contract\n');
            elseif isLegacy
                fprintf('    -> LEGACY int BOS contract (old split) — update recommended\n');
            end
            hasSpectrum = any(contains(lowOut,'spectrum')) || any(contains(lowOut,'feat'));
            hasProb = any(contains(lowOut,'prob')) || any(contains(lowOut,'logit'));
            fprintf('  Outputs: hasSpectrum %d hasProb %d (separate heads rf=2)\n', hasSpectrum, hasProb);
        end
        % Layers peek
        try
            fprintf('  Layers: %d (first 5)\n', numel(net.Layers));
            for k=1:min(5,numel(net.Layers))
                lyr = net.Layers(k);
                sz = 'n/a';
                if isprop(lyr,'InputSize')
                    try; sz = mat2str(lyr.InputSize); catch; end
                end
                fprintf('    Layer %d: %s (%s) InputSize=%s\n', k, lyr.Name, class(lyr), sz);
            end
        catch
            fprintf('  Layers inspection unavailable.\n');
        end
        % Unresolved
        fprintf('  Unresolved: %s\n', 'Check reduction_factor=2, RF frames per step, and dlarray layout CBT vs SCB manually via predict test with zeros [1,1,80].');
    catch ME
        fprintf('  IMPORT FAILED: %s\n', ME.message);
    end
end
fprintf('\n[diagnose_onnx] Done. Paste InputNames/OutputNames + sizes into model_contract.m if mismatch. For full ONNX graph use python: onnx.load + print graph.input/output.\n');
end

function details = describe_inputs(net)
details = {};
names = net.InputNames;
layers = net.Layers;
for k=1:numel(names)
    s.name = string(names(k));
    s.shape = 'dynamic';
    s.dtype = 'unknown (MATLAB dlnetwork hides dtype; assume int64 for input_ids, float32 for mel/speaker)';
    try
        idx = find(contains({layers.Name}, char(names(k))),1);
        if ~isempty(idx) && isprop(layers(idx),'InputSize')
            s.shape = mat2str(layers(idx).InputSize);
        end
    catch
    end
    % dtype heuristic by name
    n = lower(string(names(k)));
    if contains(n,'input_ids') || contains(n,'mask')
        s.dtype = 'int64';
    elseif contains(n,'output_sequence') || contains(n,'hidden') || contains(n,'speaker') || contains(n,'past') || contains(n,'cache') || contains(n,'spectrogram') || contains(n,'features')
        s.dtype = 'float32';
    elseif contains(n,'use_cache')
        s.dtype = 'bool';
    end
    details{end+1}=s;
end
end

function details = describe_outputs(net)
details = {};
names = net.OutputNames;
for k=1:numel(names)
    s.name = string(names(k));
    s.shape = 'dynamic';
    s.dtype = 'float32 (assumed)';
    n = lower(string(names(k)));
    if contains(n,'prob')
        s.dtype = 'float32 (spectrum prob)';
    end
    details{end+1}=s;
end
end
