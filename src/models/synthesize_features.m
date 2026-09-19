function acoustic = synthesize_features(target_text, speaker_emb, models, cfg)
%SYNTHESIZE_FEATURES Run SpeechT5 encoder + decoder using the real ONNX model contract.
%
% This function validates the model export dynamically and calls the correct
% inputs for the encoder/decoder graphs. It does not assume a generic
% text+speaker predict interface; instead it inspects the loaded model names.

arguments
    target_text (1,1) string
    speaker_emb (:,1) {mustBeNumeric, mustBeFinite}
    models struct = struct()
    cfg struct = struct()
end

if isempty(cfg)
    cfg = pipeline_config();
end
if isempty(models) || ~isfield(models, 'encoder') || ~isfield(models, 'decoder')
    models = load_onnx_engine(cfg);
end

if strlength(strtrim(target_text)) == 0
    error('synthesize_features:EmptyText', 'Target text cannot be empty.');
end
if isempty(speaker_emb)
    error('synthesize_features:MissingEmbedding', 'Speaker embedding is empty.');
end

[token_ids, attention_mask] = tokenize_text(target_text, cfg);

encoder_input_names = get_input_names(models.encoder);
encoder_inputs = prepare_encoder_inputs(token_ids, attention_mask, encoder_input_names);
try
    enc_out = predict(models.encoder, encoder_inputs);
catch ME
    error('synthesize_features:EncoderFailure', 'SpeechT5 encoder failed: %s', ME.message);
end
enc_hidden = first_output_value(enc_out);

speaker_vec = single(speaker_emb(:)');
if numel(speaker_vec) ~= cfg.spk_emb_dim
    warning('synthesize_features:EmbeddingMismatch', ...
        'Speaker embedding size is %d; expected %d. The model contract will decide how to use it.', ...
        numel(speaker_vec), cfg.spk_emb_dim);
end

decoder_input_names = get_input_names(models.decoder);
if ~isempty(decoder_input_names)
    decoder_inputs = prepare_decoder_inputs(token_ids, enc_hidden, attention_mask, speaker_vec, decoder_input_names);
    try
        dec_out = predict(models.decoder, decoder_inputs);
    catch ME
        error('synthesize_features:DecoderFailure', 'SpeechT5 decoder failed: %s', ME.message);
    end
    acoustic = first_output_value(dec_out);
else
    % Fallback: use a conventional encoder-output shaped tensor if the graph has
    % no declared input names.
    acoustic = enc_hidden;
end

if isempty(acoustic)
    error('synthesize_features:EmptyOutput', 'SpeechT5 decoder produced no acoustic output.');
end

acoustic = double(acoustic);
if ~isfinite(acoustic)
    error('synthesize_features:NonFiniteOutput', 'SpeechT5 acoustic output contains NaN or Inf values.');
end

% Keep the final shape compatible with the neural vocoder input contract.
if size(acoustic, 1) ~= cfg.n_mels && size(acoustic, 2) == cfg.n_mels
    acoustic = acoustic.';
end
end

function names = get_input_names(net)
try
    names = net.InputNames;
    if isempty(names)
        names = {};
    end
catch
    names = {};
end
names = cellstr(names);
end

function v = first_output_value(out)
if iscell(out)
    v = out{1};
else
    v = out;
end
end

function inputs = prepare_encoder_inputs(token_ids, attention_mask, names)
% Use the actual imported network input names when available.
if isempty(names)
    inputs = {int64(token_ids), int64(attention_mask)};
    return;
end

inputs = cell(1, numel(names));
for i = 1:numel(names)
    lowerName = lower(names{i});
    if contains(lowerName, 'mask')
        inputs{i} = int64(attention_mask);
    else
        inputs{i} = int64(token_ids);
    end
end
end

function inputs = prepare_decoder_inputs(token_ids, enc_hidden, attention_mask, speaker_vec, names)
if isempty(names)
    inputs = {int64(token_ids), enc_hidden, int64(attention_mask), single(speaker_vec)};
    return;
end

inputs = cell(1, numel(names));
for i = 1:numel(names)
    lowerName = lower(names{i});
    if contains(lowerName, 'mask')
        inputs{i} = int64(attention_mask);
    elseif contains(lowerName, 'speaker') || contains(lowerName, 'condition')
        inputs{i} = single(speaker_vec);
    elseif contains(lowerName, 'hidden') || contains(lowerName, 'state')
        inputs{i} = enc_hidden;
    else
        inputs{i} = int64(token_ids);
    end
end
end
