function setup_matlab_online()
%SETUP_MATLAB_ONLINE Prepare the repository for execution inside MATLAB Online.
%
%   Usage in MATLAB Online Command Window:
%     cd("zero-shot-voice-cloning-main")
%     run("scripts/setup_matlab_online.m")
%
%   Execution Sequence:
%     1. Resolves canonical project root across MATLAB Drive paths
%     2. Adds src/, ui/, scripts/, and tests/ to the active path
%     3. Creates models/, outputs/, and docs/ subdirectories
%     4. Verifies required MATLAB toolboxes and ONNX support package
%     5. Downloads missing ONNX assets (Xenova + openspeech, ~875.7 MB)
%     6. Validates ONNX file integrity and reports real sizes
%     7. Runs contract validation (validate_models)
%     8. Distinguishes progress across clear milestones:
%        [MILESTONE 1] Files downloaded on disk
%        [MILESTONE 2] Models imported into MATLAB dlnetwork
%        [MILESTONE 3] Static smoke inference passed
%        [MILESTONE 4] End-to-end voice cloning validated

fprintf("=================================================================\n");
fprintf("         ZERO-SHOT VOICE CLONER: MATLAB ONLINE SETUP             \n");
fprintf("=================================================================\n");

% 1. Resolve Project Root
root = pwd;
if ~isfolder(fullfile(root, 'src'))
    candidate = fileparts(fileparts(mfilename('fullpath')));
    if isfolder(fullfile(candidate, 'src'))
        root = candidate;
        cd(root);
    end
end
project_root = root;

fprintf("[setup] Project root:   %s\n", project_root);
fprintf("[setup] MATLAB release: %s\n", version);

% 2. Add Paths
addpath(genpath(fullfile(project_root, 'src')));
addpath(fullfile(project_root, 'ui'));
addpath(fullfile(project_root, 'scripts'));
addpath(fullfile(project_root, 'tests'));

% 3. Create Required Directories
required_dirs = {
    fullfile(project_root, 'models', 'speecht5'), ...
    fullfile(project_root, 'models', 'xvector'), ...
    fullfile(project_root, 'outputs'), ...
    fullfile(project_root, 'tests'), ...
    fullfile(project_root, 'docs')
};
for i = 1:numel(required_dirs)
    if ~isfolder(required_dirs{i})
        mkdir(required_dirs{i});
    end
end

% 4. Check Dependencies
fprintf("\n[setup] Checking MATLAB toolboxes and ONNX converter ...\n");
check = check_requirements();
if ~check.ok
    warning("setup_matlab_online:MissingDependencies", ...
        "Missing required dependencies: %s.\nInstall via Add-Ons before attempting inference.", ...
        strjoin(check.missing, ", "));
else
    fprintf("  Dependencies check: OK\n");
end

% 5. Verify and Acquire Model Assets
cfg = pipeline_config();
required_assets = {
    'encoder_model.onnx',           cfg.paths.encoder; ...
    'decoder_model.onnx',           cfg.paths.decoder; ...
    'decoder_with_past_model.onnx', cfg.paths.decoder_kv; ...
    'vocoder_model.onnx',           cfg.paths.vocoder; ...
    'xvector_encoder.onnx',         cfg.paths.spk_encoder ...
};

fprintf("\n[setup] Checking model files in canonical layout ...\n");
missingCount = 0;
for i = 1:size(required_assets, 1)
    target = required_assets{i, 2};
    if ~isfile(target)
        missingCount = missingCount + 1;
        fprintf("  MISSING: %s\n", target);
    else
        d = dir(target);
        fprintf("  FOUND (%.1f MB): %s\n", d.bytes / 1e6, target);
    end
end

if missingCount > 0
    fprintf("\n[setup] %d model asset(s) missing. Initiating download from Hugging Face ...\n", missingCount);
    fprintf("  Total download size is approximately 875.7 MB (fits within 20 GB MATLAB Drive quota).\n");
    download_weights();
end

% Re-verify files
stillMissing = 0;
for i = 1:size(required_assets, 1)
    if ~isfile(required_assets{i, 2})
        stillMissing = stillMissing + 1;
    end
end

% 6. Model Contract Preflight Validation
fprintf("\n[setup] Validating model contracts and imports ...\n");
report = validate_models(cfg);

% 7. Report Milestone Status
fprintf("\n=================================================================\n");
fprintf("                   SETUP MILESTONE REPORT                        \n");
fprintf("=================================================================\n");
fprintf("  [MILESTONE 1] Files on Disk:       %s\n", status_str(stillMissing == 0));
fprintf("  [MILESTONE 2] Models Imported:     %s\n", status_str(report.ok));
fprintf("  [MILESTONE 3] Smoke Inference:     PENDING (Run scripts/validate_matlab_online.m)\n");
fprintf("  [MILESTONE 4] Real Voice Cloning:  PENDING (Run VoiceClonerApp)\n");
fprintf("-----------------------------------------------------------------\n");

if ~report.ok
    fprintf("STATUS: SETUP INCOMPLETE — Do NOT launch VoiceClonerApp yet.\n");
    fprintf("Review model contract failures above.\n");
else
    fprintf("STATUS: SETUP READY FOR PREFLIGHT VALIDATION.\n");
    fprintf("Next recommended step:\n");
    fprintf("  run('scripts/validate_matlab_online.m')\n");
    fprintf("Then launch the interface:\n");
    fprintf("  VoiceClonerApp\n");
end
fprintf("=================================================================\n\n");

end

function s = status_str(b)
if b
    s = "COMPLETE (OK)";
else
    s = "INCOMPLETE";
end
end
