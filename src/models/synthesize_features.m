function acoustic = synthesize_features(target_text, speaker_emb, models, cfg)
%SYNTHESIZE_FEATURES Run SpeechT5 encoder + autoregressive decoder (verified).
%
%   acoustic = SYNTHESIZE_FEATURES(target_text, speaker_emb, models, cfg)
%
%   Verified contract (HF + Xenova):
%     1. tokenize text -> input_ids [1,T] (no BOS, EOS appended) + attention_mask [1,T]
%     2. encoder -> hidden states [1,T,768]
%     3. decoder first step: output_sequence zeros [1,1,80] + encoder states + mask + speaker_emb [1,512] -> spectrum [1,rf,80] + prob [1,rf] + past KV (rf=2)
%     4. loop: feed last predicted Mel frame [1,1,80] as output_sequence, update past KV (use_cache_branch bool if merged)
%     5. stop on prob sigmoid>0.5 per frame (after min_len) or maxlen or finite guard
%   Supports legacy split exports that expect input_ids int64 BOS (detected at runtime) as fallback.
%   reduction_factor is verified 2 (not 1) per microsoft/speecht5_tts config.json.

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
if ~isfield(cfg,'reduction_factor') || cfg.reduction_factor ~= 2
    warning('synthesize_features:ReductionFactor', 'cfg.reduction_factor=%d but verified value is 2. Using 2.', cfg.reduction_factor);
    cfg.reduction_factor = 2;
end
rf = cfg.reduction_factor;

% --- 1. Tokenize --------------------------------------------------------
[token_ids, attention_mask] = tokenize_text(target_text, cfg);
enc_input_ids = int64(token_ids);
enc_attn_mask = int64(attention_mask);

% --- 2. Encoder ---------------------------------------------------------
fprintf('[synthesize_features] Running encoder: %d tokens (ids+EOS, no BOS) \n', numel(enc_input_ids));
try
    enc_inputs = prepare_encoder_dlarrays(enc_input_ids, enc_attn_mask, models.encoder);
    enc_out = predict(models.encoder, enc_inputs);
catch ME
    error('synthesize_features:EncoderFailure', 'SpeechT5 encoder failed: %s', ME.message);
end
enc_hidden = extract_first(enc_out);
if ndims(enc_hidden)==2
    enc_hidden = reshape(enc_hidden, 1, size(enc_hidden,1), size(enc_hidden,2));
end
enc_hidden = single(enc_hidden);
if size(enc_hidden,1) ~= 1 || size(enc_hidden,3) ~= 768
    warning('synthesize_features:EncoderShape', 'Encoder hidden shape %s not [1,T,768]', mat2str(size(enc_hidden)));
end

% --- 3. Speaker conditioning --------------------------------------------
speaker_vec = single(speaker_emb(:)');
if numel(speaker_vec) ~= cfg.spk_emb_dim
    error('synthesize_features:EmbeddingDimMismatch', ...
        'Speaker embedding %d-dim does not match cfg.spk_emb_dim=%d.', ...
        numel(speaker_vec), cfg.spk_emb_dim);
end
speaker_vec = reshape(speaker_vec,1,512);
enc_mask_for_decoder = enc_attn_mask;

% --- 4. Decoder setup ---------------------------------------------------
% Initial output_sequence is all-zero Mel [1,1,80] per HF _generate_speech ("all-zero spectrum as starting token")
output_sequence = single(zeros(1,1,cfg.n_mels)); % [1,1,80]
acoustic_frames = []; % will collect [T_mel, 80]
pastKV = {};
decoderOutNames = {};
try; decoderOutNames = cellstr(models.decoder.OutputNames); catch; end

% Max steps accounting for reduction factor
maxSteps = min(cfg.max_decoder_steps, max(50, round(cfg.max_len_ratio * numel(enc_input_ids) / rf)));
% HF also has min_len guard: min_len = enc_len * min_len_ratio / rf  (here min_len_ratio 0)
minSteps = max(0, round(cfg.min_len_ratio * numel(enc_input_ids) / rf));
fprintf('[synthesize_features] Autoregressive decoding up to %d steps (rf=%d, max %d frames) minSteps %d...\n', maxSteps, rf, maxSteps*rf, minSteps);

for step = 1:maxSteps
    if step==1
        decNet = models.decoder;
        usePast = false;
    else
        if isfield(models,'decoder_kv') && ~isempty(models.decoder_kv)
            decNet = models.decoder_kv;
            usePast = true;
        else
            decNet = models.decoder;
            usePast = false;
        end
    end

    try
        if usePast
            dec_inputs = prepare_decoder_dlarrays(output_sequence, enc_hidden, enc_mask_for_decoder, speaker_vec, pastKV, decNet, true);
        else
            dec_inputs = prepare_decoder_dlarrays(output_sequence, enc_hidden, enc_mask_for_decoder, speaker_vec, {}, decNet, false);
        end
        dec_out = predict(decNet, dec_inputs);
    catch ME
        error('synthesize_features:DecoderFailure', 'Step %d decoder failed (rf=%d seq %s): %s\nHint: Check ONNX expects float output_sequence [1,1,80] not token ids. Run diagnose_onnx.', step, rf, mat2str(size(output_sequence)), ME.message);
    end

    [spectrum, prob, newPastKV] = parse_decoder_output(dec_out, decNet, rf);
    pastKV = newPastKV;

    % spectrum is [rf,80] or [1,80] if rf=1 export
    nFramesThisStep = size(spectrum,1);
    if nFramesThisStep ~= rf
        fprintf('[synthesize_features] Warning: step %d spectrum frames %d != rf %d (export may be rf=1)\n', step, nFramesThisStep, rf);
    end
    if ~all(isfinite(spectrum),'all')
        error('synthesize_features:NonFiniteFrame','Decoder produced NaN/Inf at step %d', step);
    end

    % Append frames one-by-one to respect per-frame stop
    for f = 1:nFramesThisStep
        frame = spectrum(f,:); % [1,80]
        p = [];
        if ~isempty(prob) && numel(prob) >= f
            p = prob(f);
            % prob may be logit, apply sigmoid if outside [0,1]
            if p < 0 || p > 1
                p = 1/(1+exp(-double(p)));
            end
        elseif ~isempty(prob)
            p = prob(min(f,numel(prob)));
            if p < 0 || p > 1
                p = 1/(1+exp(-double(p)));
            end
        end
        acoustic_frames = [acoustic_frames; frame]; %#ok<AGROW>
        globalFrameIdx = size(acoustic_frames,1);
        % Stop condition per frame after minSteps
        if ~isempty(p) && globalFrameIdx > minSteps * rf
            if p > cfg.stop_threshold
                fprintf('[synthesize_features] Stop prob at step %d frame %d (p=%.2f, global %d)\n', step, f, p, globalFrameIdx);
                break;
            end
        end
    end
    % Check global stop after frames appended
    if ~isempty(prob) && size(acoustic_frames,1) > minSteps*rf
        lastP = prob(min(end,nFramesThisStep));
        if lastP < 0 || lastP > 1
            lastP = 1/(1+exp(-double(lastP)));
        end
        if lastP > cfg.stop_threshold
            break;
        end
    end

    % Also handle legacy packed 81 case where prob was encoded as 81st dim (fallback)
    % parse_decoder_output already handles that by extracting last dim

    % Secondary guard: near-zero energy for 3 consecutive frames after 10
    if size(acoustic_frames,1) > 10
        recent_energy = mean(abs(acoustic_frames(end-2:end,:)),'all');
        if recent_energy < 1e-6
            fprintf('[synthesize_features] Low energy stop at step %d\n', step);
            break;
        end
    end

    % Prepare next output_sequence as last predicted frame [1,1,80]
    if ~isempty(acoustic_frames)
        last_frame = single(acoustic_frames(end,:)); % [1,80]
        output_sequence = reshape(last_frame, 1,1,80);
    end

    % Early break if prob stop triggered inside per-frame loop
    if ~isempty(prob)
        checkP = prob(min(end,nFramesThisStep));
        if checkP < 0 || checkP > 1
            checkP = 1/(1+exp(-double(checkP)));
        end
        if checkP > cfg.stop_threshold && size(acoustic_frames,1) > minSteps*rf
            break;
        end
    end
end

if isempty(acoustic_frames)
    error('synthesize_features:EmptyOutput', 'SpeechT5 decoder produced no acoustic frames.');
end

acoustic = single(acoustic_frames'); % [80, T_mel]
fprintf('[synthesize_features] Generated %d Mel frames [80 x %d] (rf=%d, steps %d)\n', size(acoustic,2), size(acoustic,1), rf, ceil(size(acoustic,2)/rf));

if ~all(isfinite(acoustic),'all')
    error('synthesize_features:NonFiniteOutput', 'Acoustic output contains NaN/Inf');
end
end

% -----------------------------------------------------------------------
function dlarrays = prepare_encoder_dlarrays(input_ids, attn_mask, net)
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

function dlarrays = prepare_decoder_dlarrays(output_sequence, enc_hidden, enc_mask, speaker_vec, pastKV, net, usePast)
% Detect contract at runtime: float mel vs legacy int BOS, plus use_cache_branch
names = cellstr(net.InputNames);
dlarrays = cell(1,numel(names));
pastIdx = 1;
nPastInputs = max(0, numel(names) - 4 - any(contains(lower(names),'use_cache')));
% Heuristic: past inputs are those containing past/key/value/cache/pkv
% But for float contract, first 4 are output_sequence, encoder_hidden_states, encoder_attention_mask, speaker_embeddings
hasUseCache = any(contains(lower(names),'use_cache'));
% Identify which name is the Mel input
isFloatContract = any(contains(lower(names),'output_sequence')) || any(contains(lower(names),'decoder_input_values')) || any(contains(lower(names),'input_values')) || any(contains(lower(names),'spectrogram'));
isLegacyInt = ~isFloatContract && any(strcmpi(names,'input_ids') | strcmpi(names,'decoder_input_ids'));
if isFloatContract && isLegacyInt
    % ambiguous — prefer float
    isLegacyInt = false;
end
if ~isFloatContract && ~isLegacyInt
    % Fallback: check dtype expectation — if first input would be int, assume legacy; else float
    % We treat any decoder that doesn't have output_sequence as legacy for backward compat
    warning('synthesize_features:UnknownDecoderInput', 'Decoder inputs [%s] do not contain output_sequence or input_ids — assuming float output_sequence contract.', strjoin(names,','));
    isFloatContract = true;
end

for i=1:numel(names)
    n = lower(strtrim(names{i}));
    if hasUseCache && contains(n,'use_cache')
        % merged model boolean flag: true when past exists
        dlarrays{i} = dlarray(logical(usePast), 'CB');
    elseif (strcmp(n,'output_sequence') || strcmp(n,'decoder_input_values') || strcmp(n,'input_values') || strcmp(n,'spectrogram')) && isFloatContract
        % float Mel [1,1,80] or [1,rf,80]? Use CBT layout as for CAM++ fbank [1,T,80]
        % Try CBT layout: reshape to [1,1,80] with CBT
        try
            dlarrays{i} = dlarray(single(output_sequence), 'CBT');
        catch
            % fallback to SCB
            dlarrays{i} = dlarray(single(reshape(output_sequence,1,80,1)), 'SCB');
        end
    elseif (strcmp(n,'input_ids') || strcmp(n,'decoder_input_ids')) && isLegacyInt
        % Legacy int BOS fallback — use BOS token 0
        bos = int64(0);
        dlarrays{i} = dlarray(bos, 'CB');
        fprintf('[synthesize_features] Legacy int input_ids contract detected at input %d (%s) — feeding BOS=0 (deprecated, update ONNX to float mel).\n', i, names{i});
    elseif strcmp(n,'encoder_hidden_states')
        dlarrays{i} = dlarray(single(enc_hidden), 'CB');
    elseif strcmp(n,'encoder_attention_mask')
        dlarrays{i} = dlarray(int64(enc_mask), 'CB');
    elseif strcmp(n,'speaker_embeddings') || strcmp(n,'speaker_embedding')
        dlarrays{i} = dlarray(single(speaker_vec), 'CB');
    elseif contains(n,'past') || contains(n,'cache') || contains(n,'present') || contains(n,'key') || contains(n,'value') || contains(n,'pkv')
        if ~isempty(pastKV) && pastIdx <= numel(pastKV)
            dlarrays{i} = pastKV{pastIdx};
            pastIdx = pastIdx+1;
        else
            if usePast
                error('synthesize_features:MissingPastKV','Decoder_with_past input "%s" requires past KV at position %d but cache empty (have %d). Step 1 should use decoder_model.onnx.', names{i}, i, numel(pastKV));
            else
                % For merged model when usePast=false, past inputs may still be expected but empty — feed dummy zero
                dlarrays{i} = dlarray(single(zeros(1,1,1)), 'CB');
            end
        end
    else
        error('synthesize_features:DecoderUnexpectedInput','Decoder unexpected input "%s" (isFloat %d). Found [%s]', names{i}, isFloatContract, strjoin(names,','));
    end
end
if numel(dlarrays)==1
    dlarrays = dlarrays{1};
end
end

function v = extract_first(out)
if iscell(out)
    v = extractdata(out{1});
else
    v = extractdata(out);
end
end

function [spectrum, prob, pastKV] = parse_decoder_output(out, net, rf)
% Decoder outputs: authoritative is spectrum [B,rf,80] and prob [B,rf] separate, plus past KV.
% Legacy may be single logits [B,1,80/81] packed.
if iscell(out)
    nOut = numel(out);
    outNames = {};
    try; outNames = lower(cellstr(net.OutputNames)); catch; end
    % Identify which output is spectrum/prob vs past
    % Strategy: first output is spectrum, second may be prob if rank 2 or contains prob/logits
    % Remaining are past KV.
    % If outNames available, use them
    spectrumRaw = [];
    probRaw = [];
    pastKV = {};
    if ~isempty(outNames)
        for k=1:nOut
            nm = outNames{k};
            if contains(nm,'prob') || contains(nm,'logit') && isempty(probRaw) && k<=2
                % second output is prob
                if isempty(spectrumRaw)
                    % first was prob? shouldn't happen — treat first as spectrum
                    spectrumRaw = extractdata(out{k});
                else
                    probRaw = extractdata(out{k});
                end
            elseif (contains(nm,'spectrum') || contains(nm,'mel') || contains(nm,'feat') || strcmp(nm,'logits')) && isempty(spectrumRaw)
                spectrumRaw = extractdata(out{k});
            elseif contains(nm,'past') || contains(nm,'present') || contains(nm,'cache') || contains(nm,'key') || contains(nm,'value')
                pastKV{end+1} = out{k}; %#ok<AGROW>
            else
                % Fallback by position: first = spectrum, second = prob, rest = past
                if isempty(spectrumRaw)
                    spectrumRaw = extractdata(out{k});
                elseif isempty(probRaw) && nOut>=2 && k==2
                    % Could be prob
                    candidate = extractdata(out{k});
                    % If candidate ndims 2 and last dim small (rf), treat as prob
                    if ndims(candidate)<=2 && size(candidate,2)<=8
                        probRaw = candidate;
                    else
                        pastKV{end+1} = out{k}; %#ok<AGROW>
                    end
                else
                    pastKV{end+1} = out{k}; %#ok<AGROW>
                end
            end
        end
        % If still ambiguous and we have 2+ outputs and second not classified, check shape
        if isempty(probRaw) && nOut>=2
            cand = extractdata(out{2});
            sz = size(cand);
            % prob is [B, rf] or [rf] ; spectrum is [B,rf,80]
            if (ndims(cand)==2 && sz(2)<=4) || (numel(cand) <= 4 && max(sz)<=4)
                probRaw = cand;
                % shift pastKV if we mis-assigned
                if ~isempty(pastKV) && numel(pastKV)>=1 && isequal(pastKV{1}, out{2})
                    pastKV = pastKV(2:end);
                end
            end
        end
    else
        % No names: heuristic by rank/size
        spectrumRaw = extractdata(out{1});
        if nOut>=2
            cand = extractdata(out{2});
            % If cand last dim ==80 or contains 80, it's spectrum; else prob or past
            if size(cand, ndims(cand))==80 || (ndims(cand)==3 && size(cand,3)==80)
                % second is not prob, it's past? shouldn't
                pastKV = out(2:end);
            else
                % check if cand looks like prob [B,rf] (rf small)
                if max(size(cand))<=8
                    probRaw = cand;
                    if nOut>2
                        pastKV = out(3:end);
                    end
                else
                    % could be past KV tensors
                    pastKV = out(2:end);
                    % try to split pastKV as cell of dlarrays
                    pastKV = cell(1,numel(out)-1);
                    for k=2:nOut
                        pastKV{k-1}=out{k};
                    end
                end
            end
        end
    end
    % Handle legacy packed 81 case
    if ~isempty(spectrumRaw) && isempty(probRaw)
        % Check if last dim is 81 (80+1 stop)
        sz = size(spectrumRaw);
        lastDim = sz(end);
        if lastDim==81
            % Split last dim as stop logit
            % spectrumRaw is [B,1,81] or [B,rf,81]
            % Extract prob as last channel
            if ndims(spectrumRaw)==3
                probRaw = squeeze(spectrumRaw(:,:,81));
                spectrumRaw = spectrumRaw(:,:,1:80);
            elseif ndims(spectrumRaw)==2
                probRaw = spectrumRaw(end);
                spectrumRaw = spectrumRaw(1:80);
            end
        end
    end
    spectrum = normalize_spectrum(spectrumRaw, rf);
    prob = normalize_prob(probRaw, rf);
    % If pastKV still empty and nOut>1 and prob not found, treat remaining as past
    if isempty(pastKV) && nOut > 1
        % Re-evaluate: if we used 1 or 2 outputs for spectrum/prob, rest is past
        used = 1 + (~isempty(probRaw));
        if nOut > used
            pastKV = cell(1, nOut-used);
            for k=used+1:nOut
                pastKV{k-used}=out{k};
            end
        end
    end
else
    spectrumRaw = extractdata(out);
    spectrum = normalize_spectrum(spectrumRaw, rf);
    prob = [];
    pastKV = {};
end
end

function s = normalize_spectrum(raw, rf)
if isempty(raw)
    s = single(zeros(0,80));
    return;
end
raw = single(raw);
% raw may be [1,1,80], [1,rf,80], [rf,80], [80,rf], [B,T,80] etc.
% Goal: return [rf_or_T,80] matrix where row = frame
sz = size(raw);
if ndims(raw)==3
    % [B,T,80] or [B,80,T]
    if sz(3)==80
        % [B,T,80] -> squeeze batch
        s = squeeze(raw);
        if ndims(s)==3
            s = reshape(s, [],80);
        elseif size(s,2)~=80 && size(s,1)==80
            s = s';
        end
        if size(s,2)~=80
            s = s';
        end
        if size(s,2)~=80
            s = reshape(raw, [],80);
        end
    elseif sz(2)==80
        % [B,80,T] -> permute
        s = permute(raw, [1,3,2]);
        s = squeeze(s);
        if size(s,2)~=80
            s = s';
        end
    else
        s = reshape(raw, [],80);
    end
elseif ndims(raw)==2
    if sz(1)==80 && sz(2)~=80
        s = raw';
    elseif sz(2)==80
        s = raw;
    elseif sz(1)<=8 && sz(2)==80
        s = raw;
    else
        % maybe [80,1] etc.
        s = reshape(raw, [],80);
    end
else
    s = reshape(raw, [],80);
end
if size(s,2)~=80
    % Try transpose
    if size(s,1)==80
        s = s';
    else
        s = reshape(raw, [],80);
    end
end
% Ensure rf handling: if s has >rf rows but rf expected 2, keep as is (multiple frames)
end

function p = normalize_prob(raw, rf)
if isempty(raw)
    p = [];
    return;
end
raw = single(raw);
% raw may be [1,rf], [rf], [1,1,rf], [B,1] etc.
p = raw(:)';
if numel(p) > rf+2
    % Could be longer sequence due to misuse; truncate to rf
    p = p(1:min(numel(p), rf));
end
end
