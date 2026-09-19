function test_tokenizer()
%TEST_TOKENIZER Validate deterministic SpeechT5 tokeniser with assertions.

import matlab.unittest.TestCase
import matlab.unittest.constraints.*
import matlab.unittest.verification.Verifiable

cfg = pipeline_config();

% Normal cases
[ids, mask] = tokenize_text("Hello world", cfg);
assert(numel(ids) == numel(mask), 'Mask length mismatch');
assert(ids(1)==cfg.bos_token_id && ids(end)==cfg.eos_token_id, 'Missing BOS/EOS');
assert(all(mask==1), 'Mask not all ones');

% Punctuation
[ids2,~] = tokenize_text("Hello, world!", cfg);
assert(numel(ids2)>5, 'Punctuation tokenisation failed');

% Whitespace collapse
[ids3,~] = tokenize_text("A  B   C", cfg);
[ids4,~] = tokenize_text("A B C", cfg);
assert(isequal(ids3,ids4), 'Whitespace not collapsed');

% Unknown char -> <unk>
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

fprintf('[test_tokenizer] All assertions passed.\n');
end
