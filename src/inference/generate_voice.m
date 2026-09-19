function result = generate_voice(referenceAudio, referenceFs, targetText, models, cfg, options)
%GENERATE_VOICE Central zero-shot voice-cloning pipeline orchestrator.
%
%   result = GENERATE_VOICE(referenceAudio, referenceFs, targetText)
%   result = GENERATE_VOICE(referenceAudio, referenceFs, targetText, models, cfg)
%   result = GENERATE_VOICE(referenceAudio, referenceFs, targetText, models, cfg, options)
%
%   End-to-End Execution Sequence:
%     1. Input Validation
%     2. DSP Preprocessing: 16 kHz mono resampling, DC removal, peak norm
%     3. Speaker Feature Extraction: 80-bin filterbank + CMN + CAM++ ONNX
%     4. Text Tokenization: SentencePiece 81 vocab, metaspace ▁, EOS only
%     5. SpeechT5 Encoder: tokens -> contextual representations (768)
%     6. SpeechT5 Decoder + KV Cache: Mel autoregression (rf=2) -> [80 x T]
%     7. Neural Vocoder: HiFi-GAN [1, 80, T] -> 16 kHz waveform
%     8. Postprocessing and Integrity Verification
%
%   Timing Integrity:
%     All stages are measured independently. Zero fabricated or duplicated timings.
%     metrics: .preprocessingTime_s, .speakerEncoderTime_s, .tokenizationTime_s,
%              .encoderTime_s, .decoderTime_s, .vocoderTime_s, .totalTime_s.

arguments
    referenceAudio (:,:) {mustBeNumeric, mustBeFinite}
    referenceFs (1,1) {mustBePositive, mustBeFinite} = 16000
    targetText (1,1) string = ""
    models struct = struct()
    cfg struct = struct()
    options.enable_denoise logical = false
end

if isempty(cfg) || ~isfield(cfg, 'fs')
    cfg = pipeline_config();
end
if isempty(models) || ~isfield(models, 'encoder')
    models = load_onnx_engine(cfg);
end

if isempty(referenceAudio)
    error("generate_voice:EmptyReferenceAudio", "Reference speech audio cannot be empty.");
end
if strlength(strtrim(targetText)) == 0
    error("generate_voice:EmptyTargetText", "Target synthesis text cannot be empty.");
end

precomputedMetrics = struct();

% --- 1. Preprocess Reference Audio --------------------------------------
t_pre = tic;
try
    [refClean, refFs, dspInfo] = preprocess_signal(referenceAudio, referenceFs, cfg.fs, ...
        enable_denoise=options.enable_denoise);
    precomputedMetrics.preprocessingTime_s = toc(t_pre);
catch ME
    error("generate_voice:PreprocessingFailed", ...
        "DSP preprocessing of reference audio failed: %s", ME.message);
end

% --- 2. Extract Speaker Embedding ---------------------------------------
t_spk = tic;
try
    speakerEmbedding = extract_speaker_embedding(refClean, refFs, models, cfg);
    precomputedMetrics.speakerEncoderTime_s = toc(t_spk);
catch ME
    error("generate_voice:SpeakerExtractionFailed", ...
        "Speaker embedding extraction via CAM++ failed: %s", ME.message);
end

% --- 3. Synthesize Voice from Embedding ---------------------------------
result = generate_voice_from_embedding(speakerEmbedding, targetText, models, cfg, precomputedMetrics);

% Attach reference metadata
result.metadata.referenceSampleRate = refFs;
result.metadata.referenceDurationSeconds = numel(refClean) / refFs;
result.metadata.dspInfo = dspInfo;

fprintf("[generate_voice] Finished: %d samples @ %d Hz (%.2f s audio) in %.2f s (RTF: %.2fx)\n", ...
    numel(result.waveform), result.sampleRate, result.metadata.audioDurationSeconds, ...
    result.metrics.totalTime_s, result.metadata.realTimeFactor);
fprintf("  Timings: pre=%.2fs | spk=%.2fs | tok=%.2fs | enc=%.2fs | dec=%.2fs | voc=%.2fs | tot=%.2fs\n", ...
    result.metrics.preprocessingTime_s, result.metrics.speakerEncoderTime_s, ...
    result.metrics.tokenizationTime_s, result.metrics.encoderTime_s, ...
    result.metrics.decoderTime_s, result.metrics.vocoderTime_s, result.metrics.totalTime_s);

end
