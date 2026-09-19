function result = check_requirements()
%CHECK_REQUIREMENTS Verify the MATLAB products and functions required by the pipeline.
%
% Returns: result.ok, result.required, result.available, result.missing

required = {
    'Audio Toolbox',
    'Deep Learning Toolbox',
    'Signal Processing Toolbox'
};

available = ver;
installed = cellfun(@(x) x(1), {available.Name}, 'UniformOutput', false);
installed = lower(installed);

required_lower = lower(required);
missing = {};
for i = 1:numel(required_lower)
    if ~any(strcmp(installed, required_lower{i}))
        missing{end + 1} = required{i};
    end
end

requiredFns = {
    'audioread',
    'audiowrite',
    'sound',
    'stft',
    'istft',
    'importONNXNetwork',
    'websave',
    'matlab.net.http',
    'dlarray'
};

for i = 1:numel(requiredFns)
    if ~exist(requiredFns{i}, 'file') && ~exist(requiredFns{i}, 'builtin')
        missing{end + 1} = requiredFns{i};
    end
end

result = struct();
result.required = required;
result.available = installed;
result.missing = missing;
result.ok = isempty(missing);
end
