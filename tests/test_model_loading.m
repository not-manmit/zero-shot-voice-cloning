function test_model_loading()
%TEST_MODEL_LOADING Validate ONNX model contracts with assertions.

cfg = pipeline_config();
required = {cfg.paths.encoder, cfg.paths.decoder, cfg.paths.decoder_kv, cfg.paths.vocoder, cfg.paths.spk_encoder};
for i=1:numel(required)
    assert(isfile(required{i}), sprintf('Required model assets are unavailable: %s', required{i}));
end

models = load_onnx_engine(cfg);
assert(isfield(models,'encoder') && ~isempty(models.encoder), 'Encoder not loaded');
assert(isfield(models,'decoder') && ~isempty(models.decoder), 'Decoder not loaded');
assert(isfield(models,'decoder_kv') && ~isempty(models.decoder_kv), 'Decoder_kv not loaded');
assert(isfield(models,'vocoder') && ~isempty(models.vocoder), 'Vocoder not loaded');
assert(isfield(models,'spk_encoder') && ~isempty(models.spk_encoder), 'Speaker encoder not loaded');

report = validate_models(cfg);
assert(report.ok, 'Model validation failed – see validate_models output');
for fn = fieldnames(report.models)'
    e = report.models.(fn{1});
    assert(e.exists, sprintf('Model %s missing', fn{1}));
    assert(e.ok, sprintf('Model %s import failed: %s', fn{1}, e.error));
    assert(~isempty(e.inputs), sprintf('Model %s has no inputs', fn{1}));
end

fprintf('[test_model_loading] All ONNX contracts validated.\n');
end
