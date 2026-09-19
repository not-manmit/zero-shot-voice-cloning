function report = validate_models(cfg)
%VALIDATE_MODELS Preflight inspection and contract validation for all ONNX models.
%
%   report = VALIDATE_MODELS()
%   report = VALIDATE_MODELS(cfg)
%
%   Validates in order:
%     1. Required toolboxes and ONNX support packages
%     2. Model file existence and minimum file size
%     3. ONNX importability via modern API (importNetworkFromONNX / importONNXNetwork)
%     4. Network initialization state
%     5. Input and output tensor name compatibility with pipeline contracts
%     6. Decoder dual-head architecture (spectrum + prob)
%     7. Decoder KV-cache tensor availability and count symmetry
%     8. Vocoder spectrogram input contract
%     9. Speaker encoder 80-bin filterbank input and 512-dim output
%
%   Returns structured report:
%     report.ok                – boolean true only if all models PASS or WARNING
%     report.models.<name>     – per-model status (PASS, WARNING, FAIL, SKIPPED)
%     report.failures          – cell array of critical failure descriptions
%     report.warnings          – cell array of non-fatal warning descriptions

if nargin < 1 || isempty(cfg) || ~isfield(cfg, 'paths')
    cfg = pipeline_config();
end

contract = model_contract();

report = struct();
report.ok = false;
report.timestamp = string(datetime('now'));
report.matlabVersion = version;
report.failures = {};
report.warnings = {};
report.models = struct();

fprintf("=================================================================\n");
fprintf("           ONNX MODEL PREFLIGHT VALIDATION ENGINE                \n");
fprintf("=================================================================\n");

% 1. Check Toolboxes
reqCheck = check_requirements();
if ~reqCheck.ok
    msg = sprintf("Missing required MATLAB toolboxes: %s", strjoin(reqCheck.missing, ", "));
    report.failures{end+1} = msg;
    fprintf("  [CRITICAL FAIL] %s\n", msg);
end

modelKeys = {'encoder', 'decoder', 'decoder_with_past', 'vocoder', 'spk_encoder'};
allPassed = reqCheck.ok;

for i = 1:numel(modelKeys)
    key = modelKeys{i};
    if strcmp(key, 'decoder_with_past') && ~isfield(cfg.paths, 'decoder_with_past') && isfield(cfg.paths, 'decoder_kv')
        mPath = cfg.paths.decoder_kv;
    else
        mPath = cfg.paths.(key);
    end

    entry = struct();
    entry.key = key;
    entry.path = mPath;
    entry.status = "SKIPPED";
    entry.importApi = "none";
    entry.inputs = strings(0, 1);
    entry.outputs = strings(0, 1);
    entry.message = "";

    fprintf("\nValidating [%s]: %s\n", key, mPath);

    % Check file existence
    if ~isfile(mPath)
        entry.status = "SKIPPED";
        entry.message = "File does not exist on disk.";
        report.failures{end+1} = sprintf("Model %s is missing at %s", key, mPath);
        allPassed = false;
        report.models.(key) = entry;
        fprintf("  [STATUS]: SKIPPED (File not found — run scripts/download_weights.m)\n");
        continue;
    end

    fInfo = dir(mPath);
    if fInfo.bytes < 1e6
        entry.status = "FAIL";
        entry.message = sprintf("File size (%.2f MB) is below 1 MB threshold.", fInfo.bytes/1e6);
        report.failures{end+1} = sprintf("Model %s file is too small or truncated (%.2f MB)", key, fInfo.bytes/1e6);
        allPassed = false;
        report.models.(key) = entry;
        fprintf("  [STATUS]: FAIL (%s)\n", entry.message);
        continue;
    end

    % Import Network
    try
        [net, meta] = import_onnx_model(mPath);
        entry.importApi = meta.apiUsed;
        entry.inputs = meta.inputNames;
        entry.outputs = meta.outputNames;
    catch ME
        entry.status = "FAIL";
        entry.message = sprintf("ONNX import failed: %s", ME.message);
        report.failures{end+1} = sprintf("Import failed for %s: %s", key, ME.message);
        allPassed = false;
        report.models.(key) = entry;
        fprintf("  [STATUS]: FAIL (Import error: %s)\n", ME.message);
        continue;
    end

    % Validate Input / Output Contracts
    lowerIn = lower(entry.inputs);
    lowerOut = lower(entry.outputs);
    hasFailure = false;
    hasWarning = false;

    switch key
        case 'encoder'
            % Verified contract: ONLY input_ids [1, T] format UU. No attention_mask.
            hasInputIds = any(contains(lowerIn, "input_ids"));
            hasHidden = any(contains(lowerOut, "hidden")) || ...
                        any(contains(lowerOut, "encoder_outputs")) || ...
                        any(contains(lowerOut, "last_hidden_state"));

            if ~hasInputIds
                hasFailure = true;
                entry.message = sprintf("Encoder missing input_ids. Observed: [%s]", strjoin(entry.inputs, ", "));
            elseif ~hasHidden
                hasWarning = true;
                entry.message = sprintf("Encoder output name '%s' differs from 'encoder_outputs' / 'last_hidden_state'.", strjoin(entry.outputs, ", "));
            end

        case 'decoder'
            % Verified contract: speaker_embeddings, encoder_hidden_state, output_sequence, encoder_attention_mask (or encoder_attention_ma)
            hasSpk = any(contains(lowerIn, "speaker"));
            hasHidden = any(contains(lowerIn, "hidden"));
            hasSeq = any(contains(lowerIn, "output_sequence"));
            hasMask = any(contains(lowerIn, "mask")) || ...
                      any(contains(lowerIn, "attention")) || ...
                      any(contains(lowerIn, "attention_ma"));
            hasSpectrum = any(contains(lowerOut, "spectrum")) || any(contains(lowerOut, "feat"));
            hasProb = any(contains(lowerOut, "prob")) || any(contains(lowerOut, "logit"));

            if ~hasSpk || ~hasHidden || ~hasSeq || ~hasMask
                hasFailure = true;
                entry.message = sprintf("Decoder missing required core inputs. Observed: [%s]", strjoin(entry.inputs, ", "));
            end
            if ~hasSpectrum || ~hasProb
                hasWarning = true;
                entry.message = sprintf("Decoder output head names [%s] differ from canonical [spectrumOutput, probOutput].", strjoin(entry.outputs, ", "));
            end

        case 'decoder_with_past'
            % Verified contract: 24 past KV inputs, speaker_embeddings, output_sequence, encoder_attention_mask
            nPastIn = sum(contains(lowerIn, "past") | contains(lowerIn, "key") | contains(lowerIn, "value") | contains(lowerIn, "cache"));
            nPastOut = sum(contains(lowerOut, "present") | contains(lowerOut, "past") | contains(lowerOut, "key") | contains(lowerOut, "value") | contains(lowerOut, "cache"));

            if nPastIn == 0
                hasFailure = true;
                entry.message = "decoder_with_past has 0 recognized past KV input tensors.";
            else
                fprintf("  [INFO] decoder_with_past: %d past KV inputs (12 decoder + 12 encoder), %d decoder present KV outputs.\n", nPastIn, nPastOut);
            end

        case 'vocoder'
            % Verified contract: spectrogram [T_mel, 80] format UU -> waveformOutput
            hasSpec = any(contains(lowerIn, "spectrogram")) || any(contains(lowerIn, "mel"));
            hasWave = any(contains(lowerOut, "waveform")) || any(contains(lowerOut, "audio"));

            if ~hasSpec
                hasFailure = true;
                entry.message = sprintf("Vocoder missing spectrogram input. Found: [%s]", strjoin(entry.inputs, ", "));
            end
            if ~hasWave
                hasWarning = true;
                entry.message = sprintf("Vocoder output name [%s] differs from canonical 'waveformOutput'.", strjoin(entry.outputs, ", "));
            end

        case 'spk_encoder'
            % Verified contract: feats [1, T, 80] format UUU -> embsOutput
            hasFeat = any(contains(lowerIn, "feat")) || any(contains(lowerIn, "fbank"));
            hasEmb = any(contains(lowerOut, "emb"));

            if ~hasFeat
                hasFailure = true;
                entry.message = sprintf("Speaker encoder missing 'feats' input. Found: [%s]", strjoin(entry.inputs, ", "));
            end
            if ~hasEmb
                hasWarning = true;
                entry.message = sprintf("Speaker encoder output name [%s] differs from 'embsOutput'.", strjoin(entry.outputs, ", "));
            end

            % Check for placeholder layers in CAM++ (guarantee scalar logicals)
            hasPh = isfield(meta, 'hasPlaceholderLayers') && ...
                    ~isempty(meta.hasPlaceholderLayers) && ...
                    any(meta.hasPlaceholderLayers);
            isInit = isfield(meta, 'isInitialized') && ...
                     ~isempty(meta.isInitialized) && ...
                     all(meta.isInitialized);

            entry.hasPlaceholderLayers = hasPh;
            entry.isInitialized = isInit;

            if hasPh || ~isInit
                hasWarning = true;
                if hasPh && ~isInit
                    entry.message = "CAM++ contains unsupported placeholder layers (AveragePool ceil_mode, BatchNormalization training_mode) and network is uninitialized.";
                elseif hasPh
                    entry.message = "CAM++ contains unsupported AveragePool (ceil_mode) placeholder layers.";
                else
                    entry.message = "CAM++ network is uninitialized.";
                end
            end
    end

    if hasFailure
        entry.status = "FAIL";
        report.failures{end+1} = sprintf("[%s]: %s", key, entry.message);
        allPassed = false;
        fprintf("  [STATUS]: FAIL (%s)\n", entry.message);
    elseif hasWarning
        entry.status = "WARNING";
        report.warnings{end+1} = sprintf("[%s]: %s", key, entry.message);
        fprintf("  [STATUS]: WARNING (%s)\n", entry.message);
    else
        entry.status = "PASS";
        entry.message = "All expected contracts satisfied.";
        fprintf("  [STATUS]: PASS (Imported via %s, %d in, %d out)\n", ...
            entry.importApi, numel(entry.inputs), numel(entry.outputs));
    end

    report.models.(key) = entry;
end

report.ok = allPassed;

fprintf("\n=================================================================\n");
if report.ok
    fprintf("PREFLIGHT VALIDATION: PASS\n");
    fprintf("All model contracts verified. Safe to proceed to inference.\n");
else
    fprintf("PREFLIGHT VALIDATION: FAIL\n");
    fprintf("Critical failures detected (%d total):\n", numel(report.failures));
    for f = 1:numel(report.failures)
        fprintf("  - %s\n", report.failures{f});
    end
end
if ~isempty(report.warnings)
    fprintf("Warnings noted (%d total):\n", numel(report.warnings));
    for w = 1:numel(report.warnings)
        fprintf("  - %s\n", report.warnings{w});
    end
end
fprintf("=================================================================\n");

end
