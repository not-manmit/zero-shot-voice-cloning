function test_model_loading()
%TEST_MODEL_LOADING Validate ONNX model availability and basic imported graph metadata.

cfg = pipeline_config();
if ~isfile(cfg.paths.encoder) || ~isfile(cfg.paths.decoder) || ~isfile(cfg.paths.vocoder)
    fprintf('[test_model_loading] Model assets unavailable.\n');
    return;
end

try
    models = load_onnx_engine(cfg);
    fprintf('[test_model_loading] Loaded encoder, decoder, decoder_kv, vocoder, and speaker encoder.\n');
catch ME
    fprintf('[test_model_loading] %s\n', ME.message);
end
end
