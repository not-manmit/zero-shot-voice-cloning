function result = check_requirements()
%CHECK_REQUIREMENTS Verify MATLAB products and functions used by the pipeline.

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

% Also check specific toolbox addon for ONNX import
requiredFns = {
    'audioread',
    'audiowrite',
    'sound',
    'stft',
    'istft',
    'hann',
    'resample',
    'importONNXNetwork',
    'websave',
    'dlarray',
    'dlnetwork',
    'predict'
};

for i = 1:numel(requiredFns)
    if ~exist(requiredFns{i}, 'file') && ~exist(requiredFns{i}, 'builtin')
        missing{end+1} = requiredFns{i}; %#ok<AGROW>
    end
end

% Special check: Deep Learning Toolbox Converter for ONNX Model Import
if ~exist('importONNXNetwork','file')
    missing{end+1} = 'Deep Learning Toolbox Converter for ONNX Model Format (Support Package)'; %#ok<AGROW>
end

result = struct();
result.required = required;
result.available = {available.Name};
result.missing = missing;
result.ok = isempty(missing);
if result.ok
    fprintf('[check_requirements] All required toolboxes/functions present.\n');
else
    fprintf('[check_requirements] Missing: %s\n', strjoin(missing,', '));
    fprintf('  Install via Home > Add-Ons > Get Add-Ons > search "ONNX".\n');
end
end
