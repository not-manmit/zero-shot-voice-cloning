function result = generate_voice(referenceAudio, referenceFs, targetText, models, cfg)
%GENERATE_VOICE End-to-end zero-shot generation orchestrator.
%
% result.waveform
% result.sampleRate
% result.speakerEmbedding
% result.acousticFeatures
% result.metrics
% result.metadata

arguments
    referenceAudio (:,:) {mustBeNumeric, mustBeFinite} = []
    referenceFs (1,1) {mustBePositive, mustBeFinite} = 16000
    targetText (1,1) string = ""
    models struct = struct()
    cfg struct = struct()
end

if isempty(cfg)
    cfg = pipeline_config();
end
if isempty(models)
    models = load_onnx_engine(cfg);
end

if isempty(referenceAudio)
    error("generate_voice:EmptyReference", "Reference audio is empty.");
end
if strlength(strtrim(targetText)) == 0
    error("generate_voice:EmptyText", "Target text cannot be empty.");
end

metrics = struct();
metrics.preprocessingTime_s = 0;
metrics.speakerEncoderTime_s = 0;
metrics.tokenizationTime_s = 0;
metrics.encoderTime_s = 0;
metrics.decoderTime_s = 0;
metrics.vocoderTime_s = 0;
metrics.totalTime_s = 0;

start_all = tic;

pre_t = tic;
[refClean, refFs] = preprocess_signal(referenceAudio, referenceFs, cfg.fs);
metrics.preprocessingTime_s = toc(pre_t);

spk_t = tic;
embedding = extract_speaker_embedding(refClean, refFs, models, cfg);
metrics.speakerEncoderTime_s = toc(spk_t);

text_t = tic;
[tokenIds, attentionMask] = tokenize_text(targetText, cfg);
metrics.tokenizationTime_s = toc(text_t);

enc_t = tic;
acoustic = synthesize_features(targetText, embedding, models, cfg);
metrics.encoderTime_s = toc(enc_t);

vocoder_t = tic;
wave = reconstruct_waveform(acoustic, cfg.fs, models, cfg);
metrics.vocoderTime_s = toc(vocoder_t);

result = struct();
result.waveform = wave;
result.sampleRate = cfg.fs;
result.speakerEmbedding = embedding;
result.acousticFeatures = acoustic;
result.metrics = metrics;
result.metadata = struct();
result.metadata.referenceSampleRate = refFs;
result.metadata.targetText = char(targetText);
result.metadata.tokenIds = tokenIds;
result.metadata.attentionMask = attentionMask;
result.metadata.modelPaths = cfg.paths;

metrics.totalTime_s = toc(start_all);
result.metrics = metrics;
end
