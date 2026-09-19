function setup_matlab_online()
%SETUP_MATLAB_ONLINE Prepare the repo for MATLAB Online model validation and app launch.
%
% Run from the project root:
%   cd("zero-shot-voice-cloning-main")
%   run("scripts/setup_matlab_online.m")

root = pwd;
if ~isfolder(fullfile(root, 'src'))
    % If the script is run from a different working directory, walk up until the repo root is found.
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
        "One or more MATLAB products are missing. Review the requirement report before running inference.");
end

cfg = pipeline_config();
required_assets = {
    'encoder_model.onnx', fullfile(cfg.paths.speecht5_root, 'encoder_model.onnx'), cfg.paths.encoder;
    'decoder_model.onnx', fullfile(cfg.paths.speecht5_root, 'decoder_model.onnx'), cfg.paths.decoder;
    'decoder_with_past_model.onnx', fullfile(cfg.paths.speecht5_root, 'decoder_with_past_model.onnx'), cfg.paths.decoder_kv;
    'vocoder_model.onnx', fullfile(cfg.paths.speecht5_root, 'vocoder_model.onnx'), cfg.paths.vocoder;
    'xvector_encoder.onnx', fullfile(cfg.paths.xvector_root, 'xvector_encoder.onnx'), cfg.paths.spk_encoder;
    'cmu_arctic_xvectors.mat', fullfile(cfg.paths.xvector_root, 'cmu_arctic_xvectors.mat'), cfg.paths.spk_embeddings;
};

missing = false;
for i = 1:size(required_assets, 1)
    target = required_assets{i, 3};
    if ~isfile(target)
        missing = true;
        fprintf("[setup_matlab_online] Missing asset: %s\n", target);
    else
        fprintf("[setup_matlab_online] Verified: %s\n", target);
    end
end

if missing
    fprintf("[setup_matlab_online] Downloading missing model assets into the canonical model directories...\n");
    download_weights();
end

report = validate_models();
fprintf("\n[setup_matlab_online] Setup summary\n");
fprintf("  Project root: %s\n", project_root);
fprintf("  Required toolboxes: %s\n", strjoin(check.required, ', '));
fprintf("  Toolbox status: %s\n", check.ok ? "OK" : "MISSING REQUIRED COMPONENTS");
fprintf("  Model validation: %s\n", report.ok ? "OK" : "FAILED");

if ~report.ok
    fprintf("\n[setup_matlab_online] Model validation failed. Review the model export contract before generation.\n");
    return;
end

fprintf("\n[setup_matlab_online] Setup finished. Launch the app with:\n");
fprintf("  VoiceClonerApp\n");
end
