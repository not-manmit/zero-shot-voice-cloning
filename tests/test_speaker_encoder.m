function test_speaker_encoder()
%TEST_SPEAKER_ENCODER Validate x-vector embedding with assertions.

cfg = pipeline_config();
if ~isfile(cfg.paths.spk_encoder)
    error('test_speaker_encoder:MissingModel', 'Required model assets are unavailable: %s', cfg.paths.spk_encoder);
end

models = load_onnx_engine(cfg);
% Synthesize 1.5 s reference (sine + noise) – deterministic
fs = cfg.fs;
t = (0:1/fs:1.5-1/fs)';
x = 0.5*sin(2*pi*120*t) + 0.1*randn(size(t));
x = x / max(abs(x));

emb = extract_speaker_embedding(x, fs, models, cfg);
assert(~isempty(emb), 'Embedding empty');
assert(numel(emb)==cfg.spk_emb_dim, 'Embedding dim mismatch');
assert(all(isfinite(emb)), 'Embedding non-finite');
assert(abs(norm(emb)-1) < 1e-4, 'Embedding not L2 normalised');
assert(~any(abs(emb)>5), 'Embedding implausible values');

% Silence should error
try
    extract_speaker_embedding(zeros(16000,1), 16000, models, cfg);
    error('test_speaker_encoder:NoError','Silent should error');
catch ME
    assert(contains(ME.identifier,'SilentAudio') || contains(ME.identifier,'TooShort'), 'Wrong error for silent');
end

fprintf('[test_speaker_encoder] OK embedding size %d L2=%.3f\n', numel(emb), norm(emb));
end
