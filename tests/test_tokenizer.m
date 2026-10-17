function test_tokenizer()
%TEST_TOKENIZER Validate verified SentencePiece-char tokeniser (Xenova 81 vocab).

cfg = pipeline_config();

% Normal case: "Hello world" -> metaspace + H e l l o per word, EOS only, case preserved, no BOS
[ids, mask] = tokenize_text("Hello world", cfg);
assert(numel(ids) == numel(mask), 'Mask length mismatch');
assert(ids(end)==cfg.eos_token_id, 'Missing EOS');
assert(ids(1) ~= cfg.bos_token_id || numel(ids)==1, 'BOS should NOT be prepended for encoder (only EOS)');
assert(all(mask==1), 'Mask not all ones');
% First token should be ▁ (id 4)
assert(ids(1)==4, 'First token should be metaspace ▁ id 4');
% Case preserved: "Hello" H is id 35, not lower h id 11
assert(any(ids==35), 'Uppercase H id 35 not found — case not preserved');
% "hello" would be h=11
[idsLower,~] = tokenize_text("hello world", cfg);
assert(any(idsLower==11) && ~any(idsLower==35), 'Lowercase h mapping broken');

% Punctuation
[ids2,~] = tokenize_text("Hello, world!", cfg);
assert(numel(ids2)>5, 'Punctuation tokenisation failed');
assert(any(ids2==23), 'Comma , id23 not found');
assert(any(ids2==44), '! id44 not found');

% Whitespace collapse
[ids3,~] = tokenize_text("A  B   C", cfg);
[ids4,~] = tokenize_text("A B C", cfg);
assert(isequal(ids3,ids4), 'Whitespace not collapsed');

% Unknown char -> <unk> 3 (e.g., © not in 81 vocab)
[ids5,~] = tokenize_text("hello ©", cfg);
assert(any(ids5==cfg.unk_token_id), 'Unknown char not mapped to unk');

% Empty -> error
try
    tokenize_text("", cfg);
    error('test_tokenizer:NoError','Expected error for empty string');
catch ME
    assert(contains(ME.identifier,'EmptyInput'), 'Wrong error for empty');
end

% Long truncation
long = repmat("a",1,500);
[idsLong,~] = tokenize_text(string(long), cfg);
assert(numel(idsLong)<=cfg.max_text_len, 'Truncation failed');
assert(idsLong(end)==cfg.eos_token_id, 'EOS not preserved after truncation');

% No lowercasing: "Hello" vs "hello" must differ
[idsH,~]=tokenize_text("Hello", cfg);
[idsL,~]=tokenize_text("hello", cfg);
assert(~isequal(idsH,idsL), 'Tokenizer should be case-sensitive');

% Target test sentence from forensic analysis
test_sent = "Hello, this is a zero-shot voice cloning demonstration in MATLAB.";
[idsSent, maskSent] = tokenize_text(test_sent, cfg);
assert(idsSent(1) == 4, 'Target sentence must begin with metaspace (ID 4)');
assert(idsSent(end) == cfg.eos_token_id, 'Target sentence must end with EOS (ID 2)');
assert(all(idsSent >= 0 & idsSent <= 80), 'All token IDs must lie within [0, 80]');
assert(numel(idsSent) == numel(maskSent), 'Attention mask dimension mismatch');
assert(numel(idsSent) >= 60, 'Expected at least 60 tokens for test sentence');

fprintf('[test_tokenizer] All assertions passed (verified 81 vocab, no BOS, case preserved, metaspace).\n');
end
