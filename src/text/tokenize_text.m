function [input_ids, attention_mask] = tokenize_text(text, cfg)
%TOKENIZE_TEXT  SpeechT5 SentencePiece-char tokeniser (verified against Xenova/microsoft).
%
%   [input_ids, attention_mask] = TOKENIZE_TEXT(text)
%   [input_ids, attention_mask] = TOKENIZE_TEXT(text, cfg)
%
%   Verified vocabulary is 81 tokens from Xenova/speecht5_tts tokenizer.json
%   (SentencePiece spm_char.model).  This replaces the earlier hand-invented
%   a-z mapping which had wrong IDs and lowercased everything.
%
%   Tokenisation procedure (matches tokenizer.json pre_tokenizer):
%   --------------------------------------------------------------
%   pre_tokenizer = [WhitespaceSplit, Metaspace(replacement="▁", add_prefix_space=true), Split]
%   So: trim, split on whitespace (collapse runs), for each word prefix "▁" then split into chars.
%   Example: "Hello world" -> ["▁","H","e","l","l","o","▁","w","o","r","l","d"] -> IDs [4,35,5,...]
%   post_processor = TemplateProcessing single: [Sequence A] + [SpecialToken </s>]  -> append EOS id 2 ONLY.
%   No BOS prepend for encoder (BOS=0 is only decoder_start_token_id for text-to-text tasks).
%   Normalizer: Precompiled (none) and normalize=false — case IS preserved.
%   Unknown chars -> <unk> id 3.
%
%   Vocabulary (81 tokens, 0-indexed) — exact from tokenizer.json:model.vocab
%   ----------------------------------
%   ID  0  <s>        ID 40 B   ID 60 K
%   ID  1  <pad>      ID 41 ?   ID 61 U
%   ID  2  </s>       ID 42 C   ID 62 V
%   ID  3  <unk>      ID 43 M   ID 63 )
%   ID  4  ▁ (U+2581) ID 44 !   ID 64 (
%   ID  5  e          ID 45 q   ID 65 Q
%   ID  6  t          ID 46 j   ID 66 Z
%   ID  7  a          ID 47 E   ID 67 ]
%   ID  8  o          ID 48 N   ID 68 [
%   ID  9  n          ID 49 P   ID 69 X
%   ID 10  i          ID 50 O   ID 70 — (U+2014)
%   ID 11  h          ID 51 D   ID 71 /
%   ID 12  s          ID 52 L   ID 72 æ (U+00E6)
%   ID 13  r          ID 53 G   ID 73 é (U+00E9)
%   ID 14  d          ID 54 R   ID 74 {
%   ID 15  l          ID 55 F   ID 75 }
%   ID 16  u          ID 56 Y   ID 76 ê (U+00EA)
%   ID 17  c          ID 57 z   ID 77 œ (U+0153)
%   ID 18  m          ID 58 J   ID 78 ̄ (U+0304)
%   ID 19  f          ID 59 :   ID 79 <mask>
%   ID 20  w          ...           ID 80 <ctc_blank>
%   ID 21  g
%   ID 22  y          (see tokenizer.json for full order)
%
%   Inputs
%   ------
%   text  – scalar string or char array
%   cfg   – (optional) struct from pipeline_config()
%
%   Outputs
%   -------
%   input_ids      – int64 row vector [1 × T]   (chars + EOS)
%   attention_mask – int64 row vector [1 × T]   (all ones)

arguments
    text  (1,1) string
    cfg   struct = struct()
end

% --- Resolve configuration -------------------------------------------
if ~isfield(cfg, 'max_text_len')
    cfg_default   = pipeline_config();
    cfg.max_text_len  = cfg_default.max_text_len;   % 450
    cfg.eos_token_id  = cfg_default.eos_token_id;   % 2
    cfg.unk_token_id  = cfg_default.unk_token_id;   % 3
    cfg.vocab_size    = cfg_default.vocab_size;      % 81
end

% --- Vocabulary (81 entries, 0-indexed, verified) --------------------
% Order = token ID. Exact order from Xenova/speecht5_tts tokenizer.json:model.vocab
% Use string array; id = index-1
VOCAB_LIST = ["<s>","<pad>","</s>","<unk>","▁","e","t","a","o","n","i","h","s","r","d","l","u","c","m","f","w","g","y",",","p","b",".","v","k",'"',"I","'","T","A","S","H",";","x","W","-","B","?","C","M","!","q","j","E","N","P","O","D","L","G","R","F","Y","z","J",":","K","U","V",")","(","Q","Z","]","[","X","—","/","æ","é","{","}","ê","œ","̄","<mask>","<ctc_blank>"];
VOCAB = VOCAB_LIST(:); % 81x1

% Build fast lookup map string->id (0-indexed)
persistent VOCAB_MAP
if isempty(VOCAB_MAP)
    VOCAB_MAP = containers.Map('KeyType','char','ValueType','double');
    for idx = 1:numel(VOCAB)
        key = char(VOCAB(idx));
        % containers.Map with char keys handles unicode via char; use string conversion
        % For special tokens like <s> we store as-is
        VOCAB_MAP(key) = idx-1;
    end
end

% --- Input validation ------------------------------------------------
if strlength(strtrim(text)) == 0
    error("tokenize_text:EmptyInput", "Input text cannot be empty.");
end

% --- Pre-tokenizer: WhitespaceSplit + Metaspace (add_prefix_space=true) -
% Trim and collapse whitespace runs to single space, then split.
text_trim = strtrim(text);
% Do NOT lower-case — case is preserved (I vs i have different IDs)
% Collapse any whitespace run to single space for WhitespaceSplit
text_trim = regexprep(text_trim, '\s+', ' ');
% Split on space
words = split(text_trim, ' ');
% Remove empty due to leading/trailing (already trimmed)
words = words(strlength(words)>0);

% Build token sequence: for each word, prepend ▁ (U+2581) then per-char (case preserved)
char_ids = int64.empty(1,0);
meta = "▁"; % U+2581 metaspace
for w = 1:numel(words)
    word = words(w);
    % First token for word is metaspace ▁
    char_ids(end+1) = lookup_id(meta, VOCAB_MAP); %#ok<AGROW>
    % Then each character of the word (split unicode-aware)
    cs = splitChars(word);
    for k = 1:numel(cs)
        c = cs(k);
        id = lookup_id(c, VOCAB_MAP);
        char_ids(end+1) = id; %#ok<AGROW>
    end
end

% Edge: if original text was single word, we already prefixed ▁; OK.
% If words empty (should not happen due to validation) fall back

% --- Add EOS only (post_processor TemplateProcessing single) -----------
eos = int64(cfg.eos_token_id);
input_ids = [char_ids, eos];   % [1 × (n_tokens + 1)]  no BOS

% --- Length check ---------------------------------------------------
max_len = cfg.max_text_len;
if numel(input_ids) > max_len
    warning("tokenize_text:TextTooLong", ...
        "Text of %d tokens exceeds max_text_len=%d. " + ...
        "Truncating to %d tokens (EOS preserved).", ...
        numel(input_ids), max_len, max_len);
    input_ids = [input_ids(1:max_len-1), eos];
end

% --- Attention mask (all ones – no padding) -------------------------
attention_mask = ones(1, numel(input_ids), "int64");

end

function ids = splitChars(s)
% Split string s into 1-char strings (unicode-aware)
% s is string scalar like "▁Hello"
chars = char(s); % may be multi-byte; better use string indexing
% Use regexp to split per character? Simple: convert to string array of characters
% MATLAB string indexing: s(k) gives char, but for unicode need extract
n = strlength(s);
ids = strings(1,n);
for i = 1:n
    ids(i) = extractBetween(s,i,i);
    if strlength(ids(i))==0
        ids(i) = string(char(s(i))); %#ok<AGROW>
    end
end
end

function id = lookup_id(c, mp)
% c is string scalar single char/token
key = char(c);
if isKey(mp, key)
    id = int64(mp(key));
else
    % Try string key
    try
        if isKey(mp, char(string(c)))
            id = int64(mp(char(string(c))));
            return;
        end
    catch
    end
    id = int64(3); % <unk>
end
end
