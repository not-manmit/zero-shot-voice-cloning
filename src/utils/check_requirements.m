function result = check_requirements()
%CHECK_REQUIREMENTS Verify MATLAB products, toolboxes, and ONNX functions.
%
%   result = CHECK_REQUIREMENTS()
%
%   Checks:
%     1. Audio Toolbox
%     2. Deep Learning Toolbox
%     3. Signal Processing Toolbox
%     4. Deep Learning Toolbox Converter for ONNX Model Format
%     5. MATLAB Release (R2023b+ minimum, R2024a+ recommended)

required = {
    'Audio Toolbox',
    'Deep Learning Toolbox',
    'Signal Processing Toolbox'
};

available = ver;
installed = lower({available.Name});
required_lower = lower(required);
missing = {};

for i = 1:numel(required_lower)
    if ~any(strcmp(installed, required_lower{i}))
        missing{end+1} = required{i}; %#ok<AGROW>
    end
end

% Check essential functions
requiredFns = {
    'audioread', ...
    'audiowrite', ...
    'sound', ...
    'stft', ...
    'istft', ...
    'hann', ...
    'resample', ...
    'websave', ...
    'dlarray', ...
    'dlnetwork', ...
    'predict'
};

for i = 1:numel(requiredFns)
    if ~exist(requiredFns{i}, 'file') && ~exist(requiredFns{i}, 'builtin')
        missing{end+1} = requiredFns{i}; %#ok<AGROW>
    end
end

% Check for either modern or legacy ONNX importer
hasModern = exist('importNetworkFromONNX', 'file') == 2 || exist('importNetworkFromONNX', 'builtin') == 5;
hasLegacy = exist('importONNXNetwork', 'file') == 2 || exist('importONNXNetwork', 'builtin') == 5;

if ~hasModern && ~hasLegacy
    missing{end+1} = 'Deep Learning Toolbox Converter for ONNX Model Format (Support Package)'; %#ok<AGROW>
end

% MATLAB release verification
v = ver('MATLAB');
relOk = true;
try
    yr = str2double(regexp(v.Release, '\d+', 'match', 'once'));
    if ~isempty(yr) && yr < 2023
        missing{end+1} = sprintf('MATLAB %s too old — requires R2023b+ (R2024a recommended)', v.Release); %#ok<AGROW>
        relOk = false;
    end
catch
end

fprintf("[check_requirements] MATLAB %s (%s)\n", v.Version, v.Release);

result = struct();
result.required = required;
result.available = {available.Name};
result.missing = missing;
result.ok = isempty(missing);

if result.ok
    fprintf("[check_requirements] All required toolboxes and ONNX functions are present.\n");
else
    fprintf("[check_requirements] Missing dependencies detected:\n");
    for m = 1:numel(missing)
        fprintf("  - %s\n", missing{m});
    end
    fprintf("Install missing components via MATLAB Home > Add-Ons > Get Add-Ons.\n");
end

end
