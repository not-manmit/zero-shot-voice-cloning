function setup_matlab_online()
%SETUP_MATLAB_ONLINE Prepare the repo for MATLAB Online validation and app launch.
%
%   Run from MATLAB Online:
%     cd("zero-shot-voice-cloning-main")
%     run("scripts/setup_matlab_online.m")

root = pwd;
if ~isfolder(fullfile(root, 'src'))
    candidate = fileparts(mfilename('fullpath'));
    for level = 1:5
        if isfolder(fullfile(candidate, 'src'))
            root = candidate;
            cd(root);
            break;
        end
        parent = fileparts(candidate);
        if strcmp(parent, candidate)
            break;
        end
        candidate = parent;
    end
end
project_root = root;
addpath(genpath(fullfile(project_root, 'src')));
addpath(fullfile(project_root, 'ui'));
addpath(fullfile(project_root, 'scripts'));

required_dirs = {
    fullfile(project_root, 'models', 'speecht5'),
    fullfile(project_root, 'models', 'xvector'),
    fullfile(project_root, 'docs'),
    fullfile(project_root, 'tests')
};
for i = 1:numel(required_dirs)
    if ~isfolder(required_dirs{i})
        mkdir(required_dirs{i});
    end
end

fprintf("[setup_matlab_online] Project root: %s\n", project_root);
fprintf("[setup_matlab_online] MATLAB version: %s\n", version);

check = check_requirements();
if ~check.ok
    warning("setup_matlab_online:MissingToolboxes", ...
        "One or more MATLAB products are missing: %s. Install before inference. See src/utils/check_requirements.m", strjoin(check.missing,', '));
else
    fprintf("[setup_matlab_online] Toolbox check: OK\n");
end

cfg = pipeline_config();
required_assets = {
    'encoder_model.onnx', cfg.paths.encoder;
    'decoder_model.onnx', cfg.paths.decoder;
    'decoder_with_past_model.onnx', cfg.paths.decoder_kv;
    'vocoder_model.onnx', cfg.paths.vocoder;
    'xvector_encoder.onnx', cfg.paths.spk_encoder;
};
fprintf('\n[setup_matlab_online] Verifying assets in canonical layout:\n');
missing = false;
for i=1:size(required_assets,1)
    target = required_assets{i,2};
    if ~isfile(target)
        missing = true;
        fprintf("  MISSING: %s\n", target);
    else
        d = dir(target);
        fprintf("  OK %.1f MB: %s\n", d.bytes/1e6, target);
    end
end

if missing
    fprintf("\n[setup_matlab_online] Downloading missing assets (Xenova + openspeech)...\n");
    fprintf("[setup_matlab_online] Note: MATLAB Drive quota 20 GB; these ~680 MB total.\n");
    download_weights();
    % Re-verify
    stillMissing=false;
    for i=1:size(required_assets,1)
        if ~isfile(required_assets{i,2})
            stillMissing=true;
            fprintf("  STILL MISSING: %s\n", required_assets{i,2});
        end
    end
    if stillMissing
        fprintf("\n[setup_matlab_online] Some assets still missing – check download_weights output and Drive quota.\n");
        fprintf("[setup_matlab_online] You can also manually download from the URLs in scripts/download_weights.m\n");
    end
end

fprintf('\n[setup_matlab_online] Validating ONNX import contracts...\n');
report = validate_models(cfg);
fprintf("\n[setup_matlab_online] Summary\n");
fprintf("  Project root: %s\n", project_root);
fprintf("  Required toolboxes: %s\n", strjoin(check.required, ', '));
if check.ok
    fprintf("  Toolbox status: OK\n");
else
    fprintf("  Toolbox status: MISSING REQUIRED COMPONENTS\n");
end
if report.ok
    fprintf("  Model validation: OK\n");
else
    fprintf("  Model validation: FAILED - see details above\n");
end

if ~report.ok
    fprintf("\n[setup_matlab_online] Validation failed. Do NOT attempt Generate until fixed.\n");
    fprintf("  Common fixes: install Deep Learning Toolbox Converter for ONNX, re-download, check opset.\n");
    return;
end

fprintf("\n[setup_matlab_online] Ready. Launch with:\n");
fprintf("  VoiceClonerApp\n");
fprintf("Then:\n");
fprintf("  validate_models   %% optional detailed check\n");
fprintf("  VoiceClonerApp    %% in Command Window\n");
end
