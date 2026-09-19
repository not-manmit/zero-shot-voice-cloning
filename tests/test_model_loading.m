function test_model_loading()
%TEST_MODEL_LOADING Validate ONNX model imports and contract assertions.

fprintf("[test_model_loading] Validating ONNX model loading and contracts ...\n");

cfg = pipeline_config();

required = {cfg.paths.encoder, cfg.paths.decoder, cfg.paths.decoder_kv, cfg.paths.vocoder, cfg.paths.spk_encoder};
for i = 1:numel(required)
    if ~isfile(required{i})
        error("test_model_loading:MissingModel", ...
            "Required model asset is unavailable: %s\nRun scripts/setup_matlab_online.m.", required{i});
    end
end

models = load_onnx_engine(cfg);
assert(isfield(models, 'encoder') && ~isempty(models.encoder), "Encoder not loaded");
assert(isfield(models, 'decoder') && ~isempty(models.decoder), "Decoder not loaded");
assert(isfield(models, 'decoder_kv') && ~isempty(models.decoder_kv), "Decoder KV not loaded");
assert(isfield(models, 'vocoder') && ~isempty(models.vocoder), "Vocoder not loaded");
assert(isfield(models, 'spk_encoder') && ~isempty(models.spk_encoder), "Speaker encoder not loaded");

report = validate_models(cfg);
assert(report.ok, sprintf("Model preflight validation failed with %d error(s)", numel(report.failures)));

fprintf("[test_model_loading] All 5 ONNX sub-models successfully loaded and verified.\n");

end
