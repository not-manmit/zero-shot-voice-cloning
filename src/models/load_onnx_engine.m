function models = load_onnx_engine(cfg, force_reload)
%LOAD_ONNX_ENGINE Load, inspect, and cache all SpeechT5 + HiFi-GAN ONNX sub-models.
%
%   models = LOAD_ONNX_ENGINE()
%   models = LOAD_ONNX_ENGINE(cfg)
%   models = LOAD_ONNX_ENGINE(cfg, force_reload)
%
%   Loads all 5 sub-models using the centralized import_onnx_model helper,
%   caches them in a persistent variable across the MATLAB session, and
%   validates basic structural availability.
%
%   Fields returned:
%     .encoder       – SpeechT5 text encoder (dlnetwork)
%     .decoder       – SpeechT5 decoder first-step (dlnetwork)
%     .decoder_kv    – SpeechT5 decoder with past KV cache (dlnetwork)
%     .vocoder       – HiFi-GAN neural vocoder (dlnetwork)
%     .spk_encoder   – CAM++ speaker embedding model (dlnetwork)
%     .contract      – Canonical model_contract() struct
%     .meta          – Per-model import forensic metadata
%     .loaded        – logical true
%     .load_time_s   – total wall-clock loading time in seconds

arguments
    cfg          struct  = struct()
    force_reload logical = false
end

persistent CACHED_MODELS;

if ~isempty(CACHED_MODELS) && isfield(CACHED_MODELS, 'loaded') && CACHED_MODELS.loaded && ~force_reload
    models = CACHED_MODELS;
    return;
end

if ~isfield(cfg, 'paths')
    cfg = pipeline_config();
end

fprintf("[load_onnx_engine] Loading SpeechT5 + CAM++ + HiFi-GAN ONNX models ...\n");
t_start = tic;

m = struct();
m.meta = struct();
m.cfg = cfg;
m.contract = model_contract();

modelKeys = {'encoder', 'decoder', 'decoder_kv', 'vocoder', 'spk_encoder'};
filePaths = {cfg.paths.encoder, cfg.paths.decoder, cfg.paths.decoder_kv, cfg.paths.vocoder, cfg.paths.spk_encoder};

for i = 1:numel(modelKeys)
    k = modelKeys{i};
    p = filePaths{i};
    fprintf("  [%s] importing: %s\n", k, p);
    try
        [net, meta] = import_onnx_model(p);
        m.(k) = net;
        m.meta.(k) = meta;
        fprintf("  [%s] OK (via %s, %d inputs, %d outputs)\n", ...
            k, meta.apiUsed, numel(meta.inputNames), numel(meta.outputNames));
    catch ME
        error("load_onnx_engine:SubmodelImportFailure", ...
            "Failed to load sub-model '%s' from %s.\nError: %s\nRun scripts/setup_matlab_online.m or scripts/download_weights.m.", ...
            k, p, ME.message);
    end
end

% Expose canonical alias so both models.decoder_kv and models.decoder_with_past work seamlessly
m.decoder_with_past = m.decoder_kv;
m.meta.decoder_with_past = m.meta.decoder_kv;

m.loaded = true;
m.load_time_s = toc(t_start);
fprintf("[load_onnx_engine] All 5 sub-models successfully imported in %.2f s.\n", m.load_time_s);

CACHED_MODELS = m;
models = CACHED_MODELS;

end
