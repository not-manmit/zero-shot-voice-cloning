function summary = validate_matlab_online()
%VALIDATE_MATLAB_ONLINE Master preflight and verification orchestrator for MATLAB Online.
%
%   summary = VALIDATE_MATLAB_ONLINE()
%
%   Workflow:
%     1. Environment & MATLAB version inspection
%     2. Toolbox and ONNX support package dependency verification
%     3. Model asset presence and file integrity checks
%     4. Forensic ONNX contract diagnostics (diagnose_onnx)
%     5. Model contract preflight validation (validate_models)
%     6. Lightweight static unit test execution (test_tokenizer, test_dsp, test_config)
%     7. Model-backed component smoke testing (test_encoder, test_speaker_encoder, test_decoder, test_vocoder)
%     8. Full synthetic pipeline integrity test (test_end_to_end)
%     9. Real human reference check (test_end_to_end_real_reference)
%
%   Outputs:
%     summary – Struct detailing the status of every milestone.
%     Prints: "MATLAB ONLINE PREFLIGHT: PASS" or "MATLAB ONLINE PREFLIGHT: FAIL"

fprintf("\n================================================================================\n");
fprintf("           MATLAB ONLINE MASTER PREFLIGHT VALIDATION ENGINE                     \n");
fprintf("================================================================================\n");

summary = struct();
summary.timestamp = string(datetime('now'));
summary.matlabVersion = version;

% 1. Environment & Paths
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(project_root, 'src')));
addpath(fullfile(project_root, 'scripts')));
addpath(fullfile(project_root, 'tests')));
addpath(fullfile(project_root, 'ui'));

fprintf("[1/6] Inspecting environment ...\n");
fprintf("  Project root:   %s\n", project_root);
fprintf("  MATLAB release: %s\n", version);

% 2. Toolboxes
fprintf("\n[2/6] Checking required toolboxes ...\n");
req = check_requirements();
summary.toolboxesOk = req.ok;
if ~req.ok
    fprintf("  [FAIL] Missing toolboxes: %s\n", strjoin(req.missing, ", "));
else
    fprintf("  [PASS] All required toolboxes and functions present.\n");
end

% 3. Model Files
fprintf("\n[3/6] Verifying ONNX model assets on MATLAB Drive ...\n");
cfg = pipeline_config();
modelFiles = {cfg.paths.encoder, cfg.paths.decoder, cfg.paths.decoder_kv, cfg.paths.vocoder, cfg.paths.spk_encoder};
allPresent = true;
for m = 1:numel(modelFiles)
    if ~isfile(modelFiles{m})
        allPresent = false;
        fprintf("  MISSING: %s\n", modelFiles{m});
    else
        fInfo = dir(modelFiles{m});
        fprintf("  OK (%.1f MB): %s\n", fInfo.bytes/1e6, modelFiles{m});
    end
end
summary.filesPresent = allPresent;

if ~allPresent
    fprintf("\n  Attempting automated model acquisition via download_weights ...\n");
    try
        download_weights();
    catch ME
        fprintf("  Download attempt failed: %s\n", ME.message);
    end
end

% 4. Model Contract Preflight Validation
fprintf("\n[4/6] Running ONNX contract validation ...\n");
modelReport = validate_models(cfg);
summary.modelsValidated = modelReport.ok;

% 5. Lightweight Static Tests (no model files required)
fprintf("\n[5/6] Running static integrity tests (tokenizer, DSP, config) ...\n");
staticOk = true;
try
    test_config();
    test_tokenizer();
    test_dsp();
    fprintf("  [PASS] Static unit tests completed successfully.\n");
catch ME
    staticOk = false;
    fprintf("  [FAIL] Static unit test failure: %s\n", ME.message);
end
summary.staticTestsOk = staticOk;

% 6. Model-Backed Smoke Tests
fprintf("\n[6/6] Running model-backed smoke tests ...\n");
modelTestsOk = false;
if summary.modelsValidated
    try
        test_model_loading();
        test_encoder();
        test_speaker_encoder();
        test_decoder();
        test_vocoder();
        test_end_to_end();
        test_end_to_end_real_reference();
        modelTestsOk = true;
        fprintf("  [PASS] Model-backed smoke tests completed successfully.\n");
    catch ME
        fprintf("  [FAIL] Model-backed test failure: %s\n", ME.message);
    end
else
    fprintf("  [SKIPPED] Model tests skipped because contract validation did not pass.\n");
end
summary.modelTestsOk = modelTestsOk;

% Final Preflight Determination
preflightPass = summary.toolboxesOk && summary.modelsValidated && summary.staticTestsOk && summary.modelTestsOk;
summary.overallPass = preflightPass;

fprintf("\n================================================================================\n");
fprintf("                       VALIDATION SUMMARY MILESTONES                            \n");
fprintf("================================================================================\n");
fprintf("  1. MATLAB Toolboxes & Add-ons:     %s\n", status_str(summary.toolboxesOk));
fprintf("  2. Model Files On Disk:            %s\n", status_str(summary.filesPresent));
fprintf("  3. Model Contract Validation:      %s\n", status_str(summary.modelsValidated));
fprintf("  4. Static Unit Tests (no models):  %s\n", status_str(summary.staticTestsOk));
fprintf("  5. Model Smoke & Pipeline Tests:   %s\n", status_str(summary.modelTestsOk));
fprintf("--------------------------------------------------------------------------------\n");

if preflightPass
    fprintf("MATLAB ONLINE PREFLIGHT: PASS\n");
    fprintf("The complete neural voice-cloning pipeline is verified and ready.\n");
    fprintf("You can launch the interactive interface with:\n");
    fprintf("  VoiceClonerApp\n");
else
    fprintf("MATLAB ONLINE PREFLIGHT: FAIL\n");
    fprintf("Action required: Review failed milestone items above before launching app.\n");
end
fprintf("================================================================================\n\n");

end

function s = status_str(val)
if val
    s = "PASS";
else
    s = "FAIL / INCOMPLETE";
end
end
