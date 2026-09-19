function result = generate_voice_from_embedding(speakerEmbedding, targetText, models, cfg, precomputedMetrics)
%GENERATE_VOICE_FROM_EMBEDDING Generate voice using an existing 512-dim speaker embedding.
%
%   result = GENERATE_VOICE_FROM_EMBEDDING(speakerEmbedding, targetText)
%   result = GENERATE_VOICE_FROM_EMBEDDING(speakerEmbedding, targetText, models, cfg)
%   result = GENERATE_VOICE_FROM_EMBEDDING(speakerEmbedding, targetText, models, cfg, precomputedMetrics)
%
%   Downstream Execution Pipeline:
%     1. Validate inputs (embedding dimensions, finiteness, text length)
%     2. Tokenize target text (SentencePiece char vocabulary, EOS only)
%     3. SpeechT5 Encoder: tokens -> hidden representations (timed independently)
%     4. SpeechT5 Decoder + KV Cache: Mel autoregression (timed independently)
%     5. HiFi-GAN Vocoder: Mel [80 x T] -> 16 kHz discrete waveform (timed independently)
%     6. Postprocess and construct comprehensive result structure
%
%   Outputs:
%     result – Struct with:
%              .waveform          – 16 kHz discrete-time audio sequence [N x 1]
%              .sampleRate        – 16 000 Hz
%              .speakerEmbedding  – 512-dimensional speaker vector [1 x 512]
%              .acousticFeatures  – Log-Mel spectrogram [80 x T]
%              .tokenIds          – Token ID row vector
%              .attentionMask     – Attention mask row vector
%              .metrics           – Exact stage execution measurements (seconds)
%              .metadata          – Synthesis parameters and diagnostics

arguments
    speakerEmbedding (:,1) {mustBeNumeric, mustBeFinite}
    targetText (1,1) string
    models struct = struct()
    cfg struct = struct()
    precomputedMetrics struct = struct()
end

if isempty(cfg) || ~isfield(cfg, 'fs')
    cfg = pipeline_config();
end
if isempty(models) || ~isfield(models, 'encoder')
    models = load_onnx_engine(cfg);
end

if isempty(speakerEmbedding) || ~all(isfinite(speakerEmbedding))
    error("generate_voice_from_embedding:InvalidEmbedding", ...
        "Speaker embedding must be a non-empty, finite numeric vector.");
end

spk_vec = single(speakerEmbedding(:)');
if numel(spk_vec) ~= cfg.spk_emb_dim
    error("generate_voice_from_embedding:DimensionMismatch", ...
        "Speaker embedding dimension %d does not match expected cfg.spk_emb_dim=%d.", ...
        numel(spk_vec), cfg.spk_emb_dim);
end

if strlength(strtrim(targetText)) == 0
    error("generate_voice_from_embedding:EmptyText", "Target text cannot be empty.");
end

t_all = tic;

% Initialize metrics
metrics = struct();
metrics.preprocessingTime_s = 0;
metrics.speakerEncoderTime_s = 0;
metrics.tokenizationTime_s = 0;
metrics.encoderTime_s = 0;
metrics.decoderTime_s = 0;
metrics.vocoderTime_s = 0;
metrics.totalTime_s = 0;

if isfield(precomputedMetrics, 'preprocessingTime_s')
    metrics.preprocessingTime_s = precomputedMetrics.preprocessingTime_s;
end
if isfield(precomputedMetrics, 'speakerEncoderTime_s')
    metrics.speakerEncoderTime_s = precomputedMetrics.speakerEncoderTime_s;
end

% --- 1. Tokenize Target Text --------------------------------------------
t_tok = tic;
[tokenIds, attentionMask] = tokenize_text(targetText, cfg);
metrics.tokenizationTime_s = toc(t_tok);

% --- 2. TTS Acoustic Synthesis (Encoder + Autoregressive Decoder) --------
try
    [acoustic, ttsTimings] = synthesize_features(targetText, spk_vec, models, cfg);
    metrics.encoderTime_s = ttsTimings.encoderTime_s;
    metrics.decoderTime_s = ttsTimings.decoderTime_s;
catch ME
    error("generate_voice_from_embedding:SynthesisFailed", ...
        "Acoustic feature synthesis failed: %s", ME.message);
end

% --- 3. Neural HiFi-GAN Vocoder -----------------------------------------
t_voc = tic;
try
    waveform = reconstruct_waveform(acoustic, cfg.fs, models, cfg);
    metrics.vocoderTime_s = toc(t_voc);
catch ME
    error("generate_voice_from_embedding:VocoderFailed", ...
        "Waveform synthesis via HiFi-GAN failed: %s", ME.message);
end

metrics.totalTime_s = toc(t_all) + metrics.preprocessingTime_s + metrics.speakerEncoderTime_s;

% --- 4. Validate and Package Result -------------------------------------
if isempty(waveform) || ~all(isfinite(waveform)) || max(abs(waveform)) < 1e-6
    error("generate_voice_from_embedding:DegenerateWaveform", ...
        "Synthesized waveform is empty, non-finite, or silent.");
end

result = struct();
result.waveform = double(waveform(:));
result.sampleRate = cfg.fs;
result.speakerEmbedding = spk_vec;
result.acousticFeatures = acoustic;
result.tokenIds = tokenIds;
result.attentionMask = attentionMask;
result.metrics = metrics;
result.metadata = struct( ...
    'targetText', char(targetText), ...
    'numTokens', numel(tokenIds), ...
    'numMelFrames', size(acoustic, 2), ...
    'audioDurationSeconds', numel(result.waveform) / result.sampleRate, ...
    'realTimeFactor', metrics.totalTime_s / max(0.01, numel(result.waveform) / result.sampleRate), ...
    'embeddingReused', (metrics.speakerEncoderTime_s == 0));

end
