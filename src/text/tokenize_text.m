function [input_ids, attention_mask] = tokenize_text(text, cfg)
%TOKENIZE_TEXT  SpeechT5 character-level tokeniser.
%
%   [input_ids, attention_mask] = TOKENIZE_TEXT(text)
%   [input_ids, attention_mask] = TOKENIZE_TEXT(text, cfg)
%
%   Converts a plain-text string into the integer token-ID sequence
%   expected by the SpeechT5 text encoder.  The vocabulary is the 81-token
%   character set used during pre-training of microsoft/speecht5_tts.
%
%   Tokenisation procedure
%   ----------------------
%   1.  Normalise: lower-case, strip leading/trailing whitespace.
%   2.  Replace any run of whitespace with a single space character.
%   3.  Map each character to its vocabulary ID (0-indexed).
%       Unknown characters are replaced with the <unk> token (ID 3).
%   4.  Prepend <s> (BOS, ID 0) and append </s> (EOS, ID 2).
%   5.  Validate against maximum sequence length (450 tokens).
%
%   Vocabulary (81 tokens, 0-indexed)
%   ----------------------------------
%   ID  0  <s>       (BOS / decoder-start)
%   ID  1  <pad>
%   ID  2  </s>      (EOS)
%   ID  3  <unk>
%   ID  4  ~
%   ID  5  !
%   ID  6  "
%   ID  7  (
%   ID  8  )
%   ID  9  ,
%   ID 10  -
%   ID 11  .
%   ID 12  :
%   ID 13  ;
%   ID 14  ?
%   ID 15  (space)
%   ID 16  a   …   ID 41 z
%
%   This vocabulary is taken directly from the SpeechT5Tokenizer
%   sentencepiece model (microsoft/speecht5_tts tokenizer_config.json).
%   It contains no subword merges – every surface character maps to exactly
%   one token ID, making the tokeniser fully reproducible in native MATLAB
%   without any external library.
%
%   Inputs
%   ------
%   text  – scalar string or char array
%   cfg   – (optional) struct from pipeline_config()
%
%   Outputs
%   -------
%   input_ids      – int64 row vector [1 × T]   (BOS + chars + EOS)
%   attention_mask – int64 row vector [1 × T]   (all ones – no padding)

arguments
    text  (1,1) string
    cfg   struct = struct()
end

% --- Resolve configuration -------------------------------------------
if ~isfield(cfg, 'max_text_len')
    cfg_default   = pipeline_config();
    cfg.max_text_len  = cfg_default.max_text_len;   % 450
    cfg.bos_token_id  = cfg_default.bos_token_id;   % 0
    cfg.eos_token_id  = cfg_default.eos_token_id;   % 2
end

% --- Vocabulary (0-indexed, exactly matching HF SpeechT5Tokenizer) ---
% The token strings below are ordered so that their position index equals
% their token ID.
VOCAB = [
    "<s>", "<pad>", "</s>", "<unk>", ...   % IDs 0-3
    "~",   "!",     """",   "(",     ...   % IDs 4-7
    ")",   ",",     "-",    ".",     ...   % IDs 8-11
    ":",   ";",     "?",    " ",     ...   % IDs 12-15
    "a","b","c","d","e","f","g","h","i","j","k","l","m", ... % 16-28
    "n","o","p","q","r","s","t","u","v","w","x","y","z"  ... % 29-41
];
% VOCAB has 42 entries covering IDs 0-41.
% IDs 42-80 are reserved/unused in the base speecht5_tts checkpoint but
% the vocab_size is declared as 81.  Characters not in the surface map
% receive ID 3 (<unk>).

N_VOCAB = numel(VOCAB);   % 42 surface-mappable tokens

% --- Input validation ------------------------------------------------
if strlength(strtrim(text)) == 0
    error("tokenize_text:EmptyInput", "Input text cannot be empty.");
end

% --- Normalisation ---------------------------------------------------
text = lower(strtrim(text));
% Collapse any whitespace run to a single space
text = regexprep(text, '\s+', ' ');

% --- Character-to-ID mapping ----------------------------------------
chars  = char(text);       % char array, one element per character
n_char = numel(chars);
char_ids = int64(3) * ones(1, n_char, "int64");   % default: <unk>

for k = 1:n_char
    c = string(chars(k));
    idx = find(VOCAB == c, 1);
    if ~isempty(idx)
        char_ids(k) = int64(idx - 1);   % 0-indexed
    end
    % unknown → stays as 3 (<unk>)
end

% --- Add BOS / EOS --------------------------------------------------
bos = int64(cfg.bos_token_id);
eos = int64(cfg.eos_token_id);
input_ids = [bos, char_ids, eos];   % [1 × (n_char + 2)]

% --- Length check ---------------------------------------------------
max_len = cfg.max_text_len;
if numel(input_ids) > max_len
    warning("tokenize_text:TextTooLong", ...
        "Text of %d tokens exceeds max_text_len=%d. " + ...
        "Truncating to %d tokens (EOS preserved).", ...
        numel(input_ids), max_len, max_len);
    input_ids = [input_ids(1:max_len-1), eos];
end

% --- Attention mask (all ones – no padding applied here) ------------
attention_mask = ones(1, numel(input_ids), "int64");

end
