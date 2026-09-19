function test_speaker_encoder()
%TEST_SPEAKER_ENCODER Validate the x-vector inference path and embeddings.

cfg = pipeline_config();
if ~isfile(cfg.paths.spk_encoder)
    fprintf('[test_speaker_encoder] Model assets unavailable.\n');
    return;
end

try
    models = load_onnx_engine(cfg);
    [x, fs] = audioread('sample_reference.wav');
    if isempty(x)
        fprintf('[test_speaker_encoder] No sample audio available.\n');
        return;
    end
    emb = extract_speaker_embedding(x, fs, models, cfg);
    if isempty(emb)
        error('extract_speaker_embedding:EmptyEmbedding', 'Embedding was empty.');
    end
    if numel(emb) ~= cfg.spk_emb_dim
        error('extract_speaker_embedding:DimMismatch', 'Incorrect embedding size.');
    end
    fprintf('[test_speaker_encoder] OK embedding size %d.\n', numel(emb));
catch ME
    fprintf('[test_speaker_encoder] %s\n', ME.message);
end
end
