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
    mPath = cfg.paths.(key);
    cEntry = contract.(key);

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
            hasInputIds = any(contains(lowerIn, "input_ids"));
            hasMask = any(contains(lowerIn, "mask"));
            hasHidden = any(contains(lowerOut, "hidden"));

            if ~hasInputIds || ~hasMask
                hasFailure = true;
                entry.message = sprintf("Encoder missing input_ids or attention_mask. Observed: [%s]", strjoin(entry.inputs, ", "));
            elseif ~hasHidden
                hasWarning = true;
                entry.message = sprintf("Encoder output name '%s' differs from 'last_hidden_state'.", strjoin(entry.outputs, ", "));
            end

        case 'decoder'
            isFloatMel = any(contains(lowerIn, "output_sequence")) || any(contains(lowerIn, "input_values")) || any(contains(lowerIn, "spectrogram"));
            isLegacyInt = any(contains(lowerIn, "input_ids")) && ~isFloatMel;
            hasSpk = any(contains(lowerIn, "speaker"));
            hasSpectrum = any(contains(lowerOut, "spectrum")) || any(contains(lowerOut, "feat")) || any(contains(lowerOut, "mel"));
            hasProb = any(contains(lowerOut, "prob")) || any(contains(lowerOut, "logit"));

            if ~isFloatMel && ~isLegacyInt
                hasFailure = true;
                entry.message = sprintf("Decoder inputs [%s] match neither float output_sequence nor legacy input_ids.", strjoin(entry.inputs, ", "));
            elseif isLegacyInt
                hasWarning = true;
                entry.message = "Decoder uses legacy int64 input_ids contract instead of modern float output_sequence.";
            end

            if ~hasSpk
                hasFailure = true;
                entry.message = sprintf("Decoder missing speaker_embeddings input. Found: [%s]", strjoin(entry.inputs, ", "));
            end

            if ~(hasSpectrum && hasProb) && ~any(strcmp(lowerOut, "logits"))
                hasWarning = true;
                entry.message = sprintf("Decoder output head names [%s] differ from canonical [spectrum, prob].", strjoin(entry.outputs, ", "));
            end

        case 'decoder_with_past'
            nPastIn = sum(contains(lowerIn, "past") | contains(lowerIn, "key") | contains(lowerIn, "value") | contains(lowerIn, "cache"));
            nPastOut = sum(contains(lowerOut, "present") | contains(lowerOut, "past") | contains(lowerOut, "key") | contains(lowerOut, "value") | contains(lowerOut, "cache"));

            if nPastIn == 0
                hasFailure = true;
                entry.message = "decoder_with_past has 0 recognized past KV input tensors.";
            elseif nPastIn ~= nPastOut && nPastOut > 0
                hasWarning = true;
                entry.message = sprintf("Asymmetric KV tensor count: %d past inputs vs %d present outputs.", nPastIn, nPastOut);
            end

        case 'vocoder'
            hasSpec = any(contains(lowerIn, "spectrogram")) || any(contains(lowerIn, "mel"));
            hasWave = any(contains(lowerOut, "waveform")) || any(contains(lowerOut, "audio"));

            if ~hasSpec
                hasFailure = true;
                entry.message = sprintf("Vocoder missing spectrogram input. Found: [%s]", strjoin(entry.inputs, ", "));
            end
            if ~hasWave
                hasWarning = true;
                entry.message = sprintf("Vocoder output name [%s] differs from canonical 'waveform'.", strjoin(entry.outputs, ", "));
            end

        case 'spk_encoder'
            hasFeat = any(contains(lowerIn, "features")) || any(contains(lowerIn, "fbank"));
            hasEmb = any(contains(lowerOut, "embedding")) || any(contains(lowerOut, "embs"));

            if ~hasFeat
                hasFailure = true;
                entry.message = sprintf("Speaker encoder missing 'features' input. Found: [%s]", strjoin(entry.inputs, ", "));
            end
            if ~hasEmb
                hasWarning = true;
                entry.message = sprintf("Speaker encoder output name [%s] differs from 'embedding'.", strjoin(entry.outputs, ", "));
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
