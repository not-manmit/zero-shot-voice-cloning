function models = load_onnx_engine(cfg, force_reload)
%LOAD_ONNX_ENGINE  Load and cache all SpeechT5 + HiFi-GAN ONNX sub-models.
%
%   models = LOAD_ONNX_ENGINE()
%   models = LOAD_ONNX_ENGINE(cfg)
%   models = LOAD_ONNX_ENGINE(cfg, force_reload)
%
%   Returns a struct with four importONNXNetwork objects and the x-vector
%   speaker encoder, loaded once and cached in a persistent variable for
%   the lifetime of the MATLAB session.  Subsequent calls return the
%   cached struct without re-loading.
%
%   Models loaded
%   -------------
%   models.encoder       – SpeechT5 text encoder (encoder_model.onnx)
%   models.decoder       – SpeechT5 decoder, first step (decoder_model.onnx)
%   models.decoder_kv    – SpeechT5 decoder, cached-KV step
%                          (decoder_with_past_model.onnx)
%   models.vocoder       – HiFi-GAN neural vocoder (vocoder_model.onnx)
%   models.spk_encoder   – SpeechBrain TDNN x-vector (xvector_encoder.onnx)
%   models.loaded        – logical true once all models are loaded
%   models.load_time_s   – wall-clock load time in seconds
%
%   Inputs
%   ------
%   cfg          – (optional) struct from pipeline_config()
%   force_reload – (optional) logical; if true, ignores cache and reloads
%                  all models.  Useful after replacing ONNX files.
%
%   Error behaviour
%   ---------------
%   Missing files → error with path and download instruction.
%   importONNXNetwork failure → error with ONNX-specific guidance.
%   Unsupported operator → error listing the operator name.

arguments
    cfg          struct  = struct()
    force_reload logical = false
end

% -----------------------------------------------------------------------
% Persistent model cache  (survives across function calls in one session)
% -----------------------------------------------------------------------
persistent CACHED_MODELS;

if ~isempty(CACHED_MODELS) && CACHED_MODELS.loaded && ~force_reload
    models = CACHED_MODELS;
    return
end

% -----------------------------------------------------------------------
% Resolve configuration
% -----------------------------------------------------------------------
if ~isfield(cfg, 'paths')
    cfg = pipeline_config();
end

fprintf("[load_onnx_engine] Loading SpeechT5 + HiFi-GAN models ...\n");
t_start = tic;

% -----------------------------------------------------------------------
% Helper to load a single ONNX file with a clear error message
% -----------------------------------------------------------------------
    function net = load_one(label, path_)
        if ~isfile(path_)
            error("load_onnx_engine:MissingFile", ...
                "Model file not found: %s\n" + ...
                "Run scripts/setup_matlab_online.m to download it.", path_);
        end
        fprintf("  [%s] importing %s ...\n", label, path_);
        try
            net = importONNXNetwork(path_, OutputLayerType="regression");
        catch ME
            % Try as a function network (preferred for ONNX with dynamic axes)
            try
                net = importONNXFunction(path_, label);
                % Wrap as dlnetwork for consistent predict() API
                net = dlnetwork(net);
            catch
                error("load_onnx_engine:ImportFailed", ...
                    "importONNXNetwork failed for %s.\n" + ...
                    "Reason: %s\n" + ...
                    "Check that Deep Learning Toolbox and its ONNX " + ...
                    "import support are installed, and that the ONNX " + ...
                    "file is valid (opset ≤ 17).", path_, ME.message);
            end
        end
        fprintf("  [%s] OK\n", label);
    end

% -----------------------------------------------------------------------
% Load all sub-models
% -----------------------------------------------------------------------
m = struct();

m.encoder    = load_one("encoder",    cfg.paths.encoder);
m.decoder    = load_one("decoder",    cfg.paths.decoder);
m.decoder_kv = load_one("decoder_kv", cfg.paths.decoder_kv);
m.vocoder    = load_one("vocoder",    cfg.paths.vocoder);
m.spk_encoder = load_one("spk_encoder", cfg.paths.spk_encoder);

% -----------------------------------------------------------------------
% Verify speaker embedding lookup table is present
% -----------------------------------------------------------------------
if isfile(cfg.paths.spk_embeddings)
    fprintf("  [spk_embeddings] loading CMU-Arctic x-vectors ...\n");
    tmp = load(cfg.paths.spk_embeddings);   % expects 'xvectors' variable
    if isfield(tmp, 'xvectors')
        m.xvectors = tmp.xvectors;   % struct with fields: speaker_id, embedding
    else
        warning("load_onnx_engine:BadEmbFile", ...
            "cmu_arctic_xvectors.mat does not contain ''xvectors'' field. " + ...
            "Re-run download_weights.m to regenerate it.");
        m.xvectors = [];
    end
else
    m.xvectors = [];
end

% -----------------------------------------------------------------------
% Finalise
% -----------------------------------------------------------------------
m.loaded       = true;
m.load_time_s  = toc(t_start);

fprintf("[load_onnx_engine] All models loaded in %.1f s\n", m.load_time_s);

CACHED_MODELS = m;
models = CACHED_MODELS;
end
