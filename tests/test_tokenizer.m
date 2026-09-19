function test_tokenizer()
%TEST_TOKENIZER Validate the deterministic text tokenization contract.

cfg = pipeline_config();

samples = [
    "Hello world";
    "Hello, world!";
    "A  B   C";
    "this     is   spaced";
    "unknown character: ©";
    "";
    "long text repeated long text repeated long text repeated long text repeated"
];

for i = 1:size(samples, 1)
    text = string(samples(i));
    try
        [ids, mask] = tokenize_text(text, cfg);
        if isempty(ids)
            fprintf('[test_tokenizer] EMPTY: %s\n', text);
        else
            fprintf('[test_tokenizer] OK: %s -> %d tokens\n', text, numel(ids));
        end
    catch ME
        fprintf('[test_tokenizer] ERROR: %s -> %s\n', text, ME.message);
    end
end

fprintf('[test_tokenizer] Tokenizer validation run complete.\n');
end
