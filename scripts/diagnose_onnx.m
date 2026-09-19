function diagnose_onnx()
%DIAGNOSE_ONNX  Print exact ONNX input/output contracts for all 5 models.
%
%   Run in MATLAB Online after setup to establish source-of-truth tensor names,
%   shapes, and dtypes. Output should be pasted into model_contract.m comments.
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
        fprintf('  MISSING — run download_weights\n');
        continue;
    end
    try
        try
            net = importONNXNetwork(p, OutputLayerType="regression", TargetNetwork="dlnetwork");
        catch
            net = importONNXNetwork(p, OutputLayerType="regression");
        end
        fprintf('  Inputs (%d):\n', numel(net.InputNames));
        for k=1:numel(net.InputNames)
            fprintf('    %d: %s\n', k, net.InputNames(k));
        end
        fprintf('  Outputs (%d):\n', numel(net.OutputNames));
        for k=1:numel(net.OutputNames)
            fprintf('    %d: %s\n', k, net.OutputNames(k));
        end
        % Try layer inspection
        try
            fprintf('  Layers: %d\n', numel(net.Layers));
            for k=1:min(5,numel(net.Layers))
                fprintf('    Layer %d: %s (%s)\n', k, net.Layers(k).Name, class(net.Layers(k)));
            end
        catch
        end
    catch ME
        fprintf('  IMPORT FAILED: %s\n', ME.message);
    end
end
fprintf('\n[diagnose_onnx] Done. Copy the InputNames/OutputNames above into model_contract.m if they differ.\n');
end
