function result = generate_voice(referenceAudio, referenceFs, targetText, models, cfg)
%GENERATE_VOICE  Central zero-shot inference orchestrator.
%
%   result = GENERATE_VOICE(referenceAudio, referenceFs, targetText)
%   result = GENERATE_VOICE(referenceAudio, referenceFs, targetText, models, cfg)
%
%   Pipeline:
%     1. validate inputs
%     2. preprocess reference -> 16 kHz mono
%     3. speaker encoder -> 512-dim L2 embedding
%     4. tokenize text -> ids + mask
%     5. TTS encoder -> hidden states
%     6. autoregressive decoder (with past KV) -> Mel [80,T]
%     7. HiFi-GAN vocoder -> waveform 16 kHz mono
%
%   Returns result struct with waveform, sampleRate, speakerEmbedding,
%   acousticFeatures [80,T], tokenIds, attentionMask, metrics, metadata.

arguments
    referenceAudio (:,:) {mustBeNumeric, mustBeFinite} = []
    referenceFs (1,1) {mustBePositive, mustBeFinite} = 16000
    targetText (1,1) string = ""
    models struct = struct()
    cfg struct = struct()
end

if isempty(cfg) || ~isfield(cfg,'fs')
    cfg = pipeline_config();
end
if isempty(models) || ~isfield(models,'encoder')
    models = load_onnx_engine(cfg);
end
if isempty(referenceAudio)
    error("generate_voice:EmptyReference", "Reference audio is empty.");
end
if strlength(strtrim(targetText)) == 0
    error("generate_voice:EmptyText", "Target text cannot be empty.");
end

metrics = struct('preprocessingTime_s',0,'speakerEncoderTime_s',0,'tokenizationTime_s',0, ...
    'encoderTime_s',0,'decoderTime_s',0,'vocoderTime_s',0,'totalTime_s',0);
t_all = tic;

% 1+2 preprocess
t = tic;
[refClean, refFs] = preprocess_signal(referenceAudio, referenceFs, cfg.fs);
metrics.preprocessingTime_s = toc(t);

% 3 speaker
t = tic;
embedding = extract_speaker_embedding(refClean, refFs, models, cfg);
metrics.speakerEncoderTime_s = toc(t);

% 4 tokenize
t = tic;
[tokenIds, attentionMask] = tokenize_text(targetText, cfg);
metrics.tokenizationTime_s = toc(t);

% 5+6 TTS: encoder + autoregressive decoder
t = tic;
acoustic = synthesize_features(targetText, embedding, models, cfg);
metrics.encoderTime_s = toc(t); % includes both encoder+decoder; split if needed
metrics.decoderTime_s = metrics.encoderTime_s; % keep for UI compatibility

% 7 vocoder
t = tic;
wave = reconstruct_waveform(acoustic, cfg.fs, models, cfg);
metrics.vocoderTime_s = toc(t);

metrics.totalTime_s = toc(t_all);

% Result
result = struct();
result.waveform = wave(:);
result.sampleRate = cfg.fs;
result.speakerEmbedding = embedding;
result.acousticFeatures = acoustic;
result.tokenIds = tokenIds;
result.attentionMask = attentionMask;
result.metrics = metrics;
result.metadata = struct('referenceSampleRate', refFs, ...
    'targetText', char(targetText), ...
    'modelPaths', cfg.paths, ...
    'genTime_s', metrics.totalTime_s);

% Post-validate waveform
if isempty(result.waveform) || ~all(isfinite(result.waveform)) || max(abs(result.waveform))<1e-6
    error('generate_voice:InvalidWaveform','Generated waveform is empty, non-finite, or silent.');
end
fprintf('[generate_voice] Done: %d samples @ %d Hz (%.2f s) total %.2f s\n', ...
    numel(result.waveform), result.sampleRate, numel(result.waveform)/result.sampleRate, metrics.totalTime_s);
end
