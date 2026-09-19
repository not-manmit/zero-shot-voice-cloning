function acoustic = synthesize_features(target_text, speaker_emb, models, cfg)
%SYNTHESIZE_FEATURES Run SpeechT5 encoder + autoregressive decoder.
%
%   acoustic = SYNTHESIZE_FEATURES(target_text, speaker_emb, models, cfg)
%
%   Implements the EXACT autoregressive loop required by SpeechT5 Xenova ONNX:
%     1. tokenize text -> input_ids [1,T] + attention_mask [1,T]
%     2. encoder -> hidden states [1,T,768]
%     3. decoder first step: input_ids=[BOS] + encoder states + mask + speaker_emb -> logits [1,1,80/81], past KV
%     4. loop: feed BOS as next decoder input_ids, update past KV (Xenova decoder keeps expecting decoder_input_ids=[BOS] + KV)
%     5. stop on EOS logit / max steps / energy guard
%   Uses both decoder_model.onnx and decoder_with_past_model.onnx when available.
%   Speaker embedding genuinely conditions every step (verified by validate_models).

arguments
    target_text (1,1) string
    speaker_emb (:,1) {mustBeNumeric, mustBeFinite}
    models struct = struct()
    cfg struct = struct()
end

if isempty(cfg) || ~isfield(cfg,'fs')
    cfg = pipeline_config();
end
if isempty(models) || ~isfield(models, 'encoder') || ~isfield(models, 'decoder')
    models = load_onnx_engine(cfg);
end

if strlength(strtrim(target_text)) == 0
    error('synthesize_features:EmptyText', 'Target text cannot be empty.');
end
if isempty(speaker_emb) || ~all(isfinite(speaker_emb))
    error('synthesize_features:MissingEmbedding', 'Speaker embedding is empty or non-finite.');
end

% --- 1. Tokenize --------------------------------------------------------
[token_ids, attention_mask] = tokenize_text(target_text, cfg);
enc_input_ids = int64(token_ids);
enc_attn_mask = int64(attention_mask);

% --- 2. Encoder ---------------------------------------------------------
fprintf('[synthesize_features] Running encoder: %d tokens\n', numel(enc_input_ids));
try
    enc_inputs = prepare_encoder_dlarrays(enc_input_ids, enc_attn_mask, models.encoder);
    enc_out = predict(models.encoder, enc_inputs);
catch ME
    error('synthesize_features:EncoderFailure', 'SpeechT5 encoder failed: %s', ME.message);
end
enc_hidden = extract_first(enc_out); % [B,T,768] or [T,768]
if ndims(enc_hidden)==2
    enc_hidden = reshape(enc_hidden, 1, size(enc_hidden,1), size(enc_hidden,2));
end
enc_hidden = single(enc_hidden);

% --- 3. Speaker conditioning --------------------------------------------
speaker_vec = single(speaker_emb(:)'); % [1,512] row
if numel(speaker_vec) ~= cfg.spk_emb_dim
    error('synthesize_features:EmbeddingDimMismatch', ...
        'Speaker embedding %d-dim does not match cfg.spk_emb_dim=%d. Model graph would mis-condition.', ...
        numel(speaker_vec), cfg.spk_emb_dim);
end
speaker_vec = reshape(speaker_vec,1,512);

enc_mask_for_decoder = enc_attn_mask;

% --- 4. Autoregressive decoder loop -------------------------------------
decoder_input_id = int64(cfg.bos_token_id); % BOS=0, Xenova decoder expects this every step with KV

acoustic_frames = []; % will collect [T_mel, 80]
pastKV = {};
% Capture KV name mapping for explicit routing diagnostics
decoderPastInputNames = {};
if isfield(models,'decoder_kv') && ~isempty(models.decoder_kv)
    try; decoderPastInputNames = cellstr(models.decoder_kv.InputNames); catch; end
end
decoderOutNames = {};
try; decoderOutNames = cellstr(models.decoder.OutputNames); catch; end

maxSteps = min(cfg.max_decoder_steps, max(50, round(cfg.max_len_ratio * numel(enc_input_ids))));
fprintf('[synthesize_features] Autoregressive decoding up to %d steps...\n', maxSteps);

for step = 1:maxSteps
    if step==1
        decNet = models.decoder;
        usePastFlag = false;
    else
        if isfield(models,'decoder_kv') && ~isempty(models.decoder_kv)
            decNet = models.decoder_kv;
            usePastFlag = true;
        else
            decNet = models.decoder;
            usePastFlag = false;
        end
    end

    try
        if usePastFlag
            dec_inputs = prepare_decoder_with_past_dlarrays(decoder_input_id, enc_hidden, enc_mask_for_decoder, speaker_vec, pastKV, decNet);
        else
            dec_inputs = prepare_decoder_dlarrays(decoder_input_id, enc_hidden, enc_mask_for_decoder, speaker_vec, decNet);
        end
        dec_out = predict(decNet, dec_inputs);
    catch ME
        error('synthesize_features:DecoderFailure', 'Step %d decoder failed: %s', step, ME.message);
    end

    [logits, newPastKV] = parse_decoder_output(dec_out, decoderOutNames);
    logits = single(logits);
    if ndims(logits)==3
        frame = squeeze(logits); % [80] or [1,80]
        frame = frame(:)';
    elseif ndims(logits)==2
        frame = logits(:)';
    else
        frame = squeeze(logits)';
    end
    % If model emits 81 (80 mel + 1 stop), split stop logit
    stop_logit = [];
    if numel(frame) == 81
        stop_logit = frame(end);
        frame = frame(1:80);
    elseif numel(frame) == 80
        % no stop logit
    else
        error('synthesize_features:UnexpectedLogitDim','Decoder output dim %d not 80 or 81 (logits size %s)', numel(frame), mat2str(size(logits)));
    end

    if ~all(isfinite(frame))
        error('synthesize_features:NonFiniteFrame','Decoder produced NaN/Inf at step %d', step);
    end

    acoustic_frames = [acoustic_frames; frame]; %#ok<AGROW>
    pastKV = newPastKV;

    % Stop condition: stop logit > threshold
    if ~isempty(stop_logit)
        p_stop = 1/(1+exp(-double(stop_logit)));
        if p_stop > cfg.stop_threshold
            fprintf('[synthesize_features] Stop token at step %d (p=%.2f)\n', step, p_stop);
            break;
        end
    end

    % Secondary stop: near-zero energy for 3 consecutive steps
    if step > 10
        recent_energy = mean(abs(acoustic_frames(end-2:end,:)),'all');
        if recent_energy < 1e-6
            fprintf('[synthesize_features] Low energy stop at step %d\n', step);
            break;
        end
    end
end

if isempty(acoustic_frames)
    error('synthesize_features:EmptyOutput', 'SpeechT5 decoder produced no acoustic frames.');
end

% Acoustic is [T_mel, 80] -> [80, T_mel] for vocoder contract
acoustic = single(acoustic_frames'); % [80, T_mel]
fprintf('[synthesize_features] Generated %d Mel frames [80 x %d]\n', size(acoustic,2), size(acoustic,1));

if ~all(isfinite(acoustic),'all')
    error('synthesize_features:NonFiniteOutput', 'Acoustic output contains NaN/Inf');
end
end

% -----------------------------------------------------------------------
function dlarrays = prepare_encoder_dlarrays(input_ids, attn_mask, net)
% Explicit contract: input_ids [1,T] int64, attention_mask [1,T] int64
% No heuristic fallback — exact name match or error.
names = cellstr(net.InputNames);
if numel(names) ~= 2
    error('synthesize_features:EncoderContract','Encoder expected 2 inputs [input_ids,attention_mask] but found %d [%s]', numel(names), strjoin(names,','));
end
dlarrays = cell(1,numel(names));
for i=1:numel(names)
    n = lower(strtrim(names{i}));
    if strcmp(n,'input_ids')
        dlarrays{i} = dlarray(int64(input_ids), 'CB');
    elseif strcmp(n,'attention_mask')
        dlarrays{i} = dlarray(int64(attn_mask), 'CB');
    else
        error('synthesize_features:EncoderUnexpectedInput','Encoder unexpected input name "%s" (expected input_ids/attention_mask)', names{i});
    end
end
if numel(dlarrays)==1
    dlarrays = dlarrays{1};
end
end

function dlarrays = prepare_decoder_dlarrays(decoder_id, enc_hidden, enc_mask, speaker_vec, net)
% Explicit contract: 4 inputs, exact names. Error on unknown.
names = cellstr(net.InputNames);
% Xenova decoder_model.onnx has exactly 4 inputs
if numel(names) < 4
    error('synthesize_features:DecoderContract','Decoder expected 4 inputs but found %d [%s]', numel(names), strjoin(names,','));
end
dlarrays = cell(1,numel(names));
for i=1:numel(names)
    n = lower(strtrim(names{i}));
    if strcmp(n,'input_ids') || strcmp(n,'decoder_input_ids')
        dlarrays{i} = dlarray(int64(decoder_id(:)'), 'CB');
    elseif strcmp(n,'encoder_hidden_states')
        dlarrays{i} = dlarray(single(enc_hidden), 'CB');
    elseif strcmp(n,'encoder_attention_mask')
        dlarrays{i} = dlarray(int64(enc_mask), 'CB');
    elseif strcmp(n,'speaker_embeddings') || strcmp(n,'speaker_embedding')
        dlarrays{i} = dlarray(single(speaker_vec), 'CB');
    else
        error('synthesize_features:DecoderUnexpectedInput','Decoder unexpected input "%s" (expected input_ids/encoder_hidden_states/encoder_attention_mask/speaker_embeddings) Found [%s]', names{i}, strjoin(names,','));
    end
end
if numel(dlarrays)==1
    dlarrays = dlarrays{1};
end
end

function dlarrays = prepare_decoder_with_past_dlarrays(decoder_id, enc_hidden, enc_mask, speaker_vec, pastKV, net)
% Explicit mapping: first 4 are same as decoder, rest are past_key_values.* in order.
% We rely on output->input order coincidence validated by validate_models (count check).
% Unknown name -> error, not silent fallback.
names = cellstr(net.InputNames);
dlarrays = cell(1,numel(names));
pastIdx = 1;
nPastInputs = max(0, numel(names)-4);
if ~isempty(pastKV) && numel(pastKV) ~= nPastInputs && nPastInputs > 0
    fprintf('[synthesize_features] Warning: pastKV count %d != expected %d past inputs\n', numel(pastKV), nPastInputs);
end
for i=1:numel(names)
    n = lower(strtrim(names{i}));
    if strcmp(n,'input_ids') || strcmp(n,'decoder_input_ids')
        dlarrays{i} = dlarray(int64(decoder_id(:)'), 'CB');
    elseif strcmp(n,'encoder_hidden_states')
        dlarrays{i} = dlarray(single(enc_hidden), 'CB');
    elseif strcmp(n,'encoder_attention_mask')
        dlarrays{i} = dlarray(int64(enc_mask), 'CB');
    elseif strcmp(n,'speaker_embeddings') || strcmp(n,'speaker_embedding')
        dlarrays{i} = dlarray(single(speaker_vec), 'CB');
    elseif contains(n,'past') || contains(n,'key') || contains(n,'value') || contains(n,'cache') || contains(n,'pkv')
        % Past KV tensors — explicit positional mapping
        if ~isempty(pastKV) && pastIdx <= numel(pastKV)
            dlarrays{i} = pastKV{pastIdx};
            pastIdx = pastIdx+1;
        else
            % First step after switch with empty cache should not happen, but guard
            error('synthesize_features:MissingPastKV','Decoder_with_past input "%s" requires past KV at position %d but cache is empty/insufficient (have %d). This indicates step 1 should use decoder_model.onnx.', names{i}, i, numel(pastKV));
        end
    else
        error('synthesize_features:DecoderKvUnexpectedInput','Decoder_with_past unexpected input "%s" Found [%s]', names{i}, strjoin(names,','));
    end
end
end

function v = extract_first(out)
if iscell(out)
    v = extractdata(out{1});
else
    v = extractdata(out);
end
end

function [logits, pastKV] = parse_decoder_output(out, outNames)
% Decoder outputs: first is logits, rest are past KV tensors per layer.
% Explicit: logits = outputs{1}, past = outputs{2:end} in order.
if iscell(out)
    logits = extractdata(out{1});
    pastKV = cell(1, numel(out)-1);
    for k=2:numel(out)
        pastKV{k-1} = out{k}; % keep as dlarray for next step (preserve dims)
    end
else
    logits = extractdata(out);
    pastKV = {};
end
end
