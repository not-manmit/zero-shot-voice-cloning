function [acoustic, timings] = synthesize_features(target_text, speaker_emb, models, cfg)
%SYNTHESIZE_FEATURES Run SpeechT5 encoder + autoregressive decoder with KV caching.
%
%   [acoustic, timings] = SYNTHESIZE_FEATURES(target_text, speaker_emb, models, cfg)
%
%   Execution Architecture:
%     1. Tokenize text: SentencePiece 81-token char vocabulary (EOS appended, no BOS)
%     2. Encoder Forward Pass: text tokens -> encoder hidden states [1, T, 768] (TIMED SEPARATELY)
%     3. Decoder Step 1: zero Mel frame [1, 1, 80] + encoder context + speaker embedding
%        -> predicted Mel [rf, 80] + stop probabilities + initial KV cache tensors
%     4. Autoregressive Loop (Step 2 .. maxSteps):
%        feed last predicted Mel frame [1, 1, 80] + previous KV cache -> next Mel frames + updated KV cache
%     5. Stop criteria evaluation:
%        - Sigmoid probability threshold (after minSteps * rf frames)
%        - Consecutive stop frames required (prevents premature truncation)
%        - Numerical bounds (NaN / Inf guards)
%        - Stagnation / energy collapse detection
%        - Hard maximum frame limit
%
%   CRITICAL DESIGN DECISIONS:
%     - The PRE-POSTNET output (spectrumOutput) feeds back as input to the next
%       decoder step (autoregressive loop input).
%     - The POSTNET output (output_sequence_out) is accumulated for the final
%       Mel spectrogram passed to HiFi-GAN, because SpeechT5's 5-layer Conv1D
%       Postnet adds residual harmonic refinement that produces smoother Mel frames.
%     - min_len_ratio is set to 3.0 (not 0.0!) to prevent premature stop-token
%       truncation. For English TTS at 16 kHz (hop=256, ~16 ms/frame), each text
%       token corresponds to ~3.5-5.0 Mel frames, so a 66-token sentence should
%       produce ~200-300 Mel frames (~3.2-4.8 seconds), NOT 69 frames.
%
%   Outputs:
%     acoustic – [80 x T_mel] single-precision log-Mel feature matrix
%     timings  – struct with separate .encoderTime_s and .decoderTime_s measurements

arguments
    target_text (1,1) string
    speaker_emb (:,1) {mustBeNumeric, mustBeFinite}
    models struct = struct()
    cfg struct = struct()
end

if isempty(cfg) || ~isfield(cfg, 'fs')
    cfg = pipeline_config();
end
if isempty(models) || ~isfield(models, 'encoder') || ~isfield(models, 'decoder')
    models = load_onnx_engine(cfg);
end

if strlength(strtrim(target_text)) == 0
    error("synthesize_features:EmptyText", "Target synthesis text cannot be empty.");
end
if isempty(speaker_emb) || ~all(isfinite(speaker_emb))
    error("synthesize_features:InvalidSpeakerEmbedding", "Speaker embedding is empty or contains non-finite values.");
end

rf = cfg.reduction_factor;
if isempty(rf) || rf < 1
    rf = 2; % Canonical SpeechT5 reduction factor
end

timings = struct('encoderTime_s', 0, 'decoderTime_s', 0);

% --- 1. Tokenize Text ---------------------------------------------------
[token_ids, attention_mask] = tokenize_text(target_text, cfg);
enc_input_ids = int64(token_ids);
enc_attn_mask = int64(attention_mask);
T_tokens = numel(enc_input_ids);

% --- 2. Encoder Forward Pass (TIMED INDEPENDENTLY) ----------------------
t_enc = tic;
try
    enc_inputs = tensor_contract_utils.format_encoder_inputs(enc_input_ids, models.encoder);
    [enc_out, models.encoder] = tensor_contract_utils.predict_net(models.encoder, enc_inputs);
    enc_hidden = tensor_contract_utils.parse_encoder_outputs(enc_out, models.encoder);
catch ME
    error("synthesize_features:EncoderFailed", ...
        "SpeechT5 encoder forward pass failed.\nError: %s\nInputs: input_ids [%d tokens].", ...
        ME.message, T_tokens);
end
timings.encoderTime_s = toc(t_enc);

% --- 3. Speaker Conditioning Vector -------------------------------------
spk_vec = single(speaker_emb(:)');
if numel(spk_vec) ~= cfg.spk_emb_dim
    error("synthesize_features:SpeakerDimMismatch", ...
        "Speaker embedding dimension %d does not match expected %d.", numel(spk_vec), cfg.spk_emb_dim);
end

% --- 4. Autoregressive Decoder Loop (TIMED INDEPENDENTLY) ---------------
t_dec = tic;

% Starting token: all-zero Mel frame [1, 1, 80] per SpeechT5 specification
output_sequence = single(zeros(1, 1, cfg.n_mels));
postnet_frames = [];    % Accumulated POSTNET Mel frames for final output -> HiFi-GAN
prepostnet_frames = []; % Accumulated pre-postnet frames (for diagnostics / fallback)
kv_cache = struct();

% Step calculation
max_steps = min(cfg.max_decoder_steps, max(30, round(cfg.max_len_ratio * T_tokens / rf)));

% CRITICAL FIX: Use min_len_ratio = 3.0 if config has 0.0 (prevents premature
% stop-token truncation that caused 1.10s / 69 frames of gibberish).
% For English TTS: ~3.5-5.0 Mel frames per text token, so min_len_ratio = 3.0
% means min_steps = ceil(3.0 * T_tokens / rf) decoder steps before stop is allowed.
effective_min_len_ratio = cfg.min_len_ratio;
if effective_min_len_ratio < 1.0
    effective_min_len_ratio = 3.0;
    fprintf("[synthesize_features] Overriding min_len_ratio from %.1f to %.1f (prevents premature stop truncation)\n", ...
        cfg.min_len_ratio, effective_min_len_ratio);
end
min_steps = max(1, round(effective_min_len_ratio * T_tokens / rf));

% Require consecutive stop frames to prevent spurious single-frame stop triggers
consecutive_stop_required = 2;
consecutive_stop_count = 0;

dec_kv_net = [];
if isfield(models, 'decoder_with_past') && ~isempty(models.decoder_with_past)
    dec_kv_net = models.decoder_with_past;
elseif isfield(models, 'decoder_kv') && ~isempty(models.decoder_kv)
    dec_kv_net = models.decoder_kv;
end
has_kv_model = ~isempty(dec_kv_net);

for step = 1:max_steps
    use_past = (step > 1) && has_kv_model && ~isempty(fieldnames(kv_cache));

    if use_past
        current_net = dec_kv_net;
    else
        current_net = models.decoder;
    end

    % Format decoder inputs
    try
        dec_inputs = tensor_contract_utils.format_decoder_inputs( ...
            output_sequence, enc_hidden, enc_attn_mask, spk_vec, kv_cache, current_net, use_past);
        [dec_raw_out, current_net] = tensor_contract_utils.predict_net(current_net, dec_inputs);
        if use_past
            dec_kv_net = current_net;
        else
            models.decoder = current_net;
        end
    catch ME
        error("synthesize_features:DecoderStepFailed", ...
            "SpeechT5 decoder failed at autoregressive step %d (usePast=%s, rf=%d):\nError: %s\nCheck models/speecht5 ONNX contract via scripts/diagnose_onnx.m.", ...
            step, string(use_past), rf, ME.message);
    end

    % Parse outputs: pre-postnet spectrum [rf x 80], postnet spectrum [rf x 80],
    % calibrated stop probabilities [1 x rf], new KV cache
    [spectrum_frames, stop_probs, ~, kv_cache, postnet_spectrum_frames] = ...
        tensor_contract_utils.parse_decoder_outputs(dec_raw_out, current_net, rf, kv_cache);

    % Verify numerical finiteness
    if ~all(isfinite(spectrum_frames), 'all')
        warning("synthesize_features:NonFiniteDecoderFrame", ...
            "Decoder produced non-finite values at step %d. Clamping to zero.", step);
        spectrum_frames(~isfinite(spectrum_frames)) = 0;
    end
    if ~all(isfinite(postnet_spectrum_frames), 'all')
        postnet_spectrum_frames(~isfinite(postnet_spectrum_frames)) = 0;
    end

    n_frames_step = size(spectrum_frames, 1);
    stop_triggered = false;

    % Append frames and evaluate stop condition per frame
    for f = 1:n_frames_step
        % Accumulate postnet frames for final output
        postnet_frame = postnet_spectrum_frames(f, :);
        postnet_frames = [postnet_frames; postnet_frame]; %#ok<AGROW>

        % Accumulate pre-postnet frames (for autoregressive feedback and diagnostics)
        prepostnet_frame = spectrum_frames(f, :);
        prepostnet_frames = [prepostnet_frames; prepostnet_frame]; %#ok<AGROW>

        curr_total_frames = size(postnet_frames, 1);

        % Stop condition: only after minimum steps, and require consecutive stop frames
        if curr_total_frames > (min_steps * rf) && numel(stop_probs) >= f
            prob_f = stop_probs(f);
            if prob_f > cfg.stop_threshold
                consecutive_stop_count = consecutive_stop_count + 1;
                if consecutive_stop_count >= consecutive_stop_required
                    stop_triggered = true;
                    break;
                end
            else
                consecutive_stop_count = 0;
            end
        end
    end

    if stop_triggered
        break;
    end

    % Stagnation / energy collapse guard (check last 5 frames after 15 frames)
    if size(postnet_frames, 1) >= 15
        recent_energy = mean(abs(postnet_frames(end-4:end, :)), 'all');
        if recent_energy < 1e-7
            break;
        end
    end

    % Prepare next output_sequence: last predicted PRE-POSTNET Mel frame [1, 1, 80]
    % CRITICAL: The autoregressive decoder feedback MUST use the pre-postnet spectrum
    % (spectrumOutput), NOT the postnet output. The decoder prenet expects raw decoder
    % output for correct positional encoding and attention alignment.
    last_frame = prepostnet_frames(end, :);
    output_sequence = reshape(single(last_frame), 1, 1, cfg.n_mels);
end

timings.decoderTime_s = toc(t_dec);

if isempty(postnet_frames)
    error("synthesize_features:EmptyAcousticFeatures", ...
        "SpeechT5 decoder finished without producing acoustic frames.");
end

% Return standard [n_mels x T_mel] matrix using POSTNET frames
acoustic = single(postnet_frames');

fprintf("[synthesize_features] Generated %d Mel frames (%d decoder steps, %.2f s audio at 16 kHz)\n", ...
    size(acoustic, 2), step, size(acoustic, 2) * 256 / 16000);

if ~all(isfinite(acoustic), 'all')
    error("synthesize_features:NonFiniteAcousticMatrix", ...
        "Generated acoustic Mel matrix contains NaN or Inf values.");
end

end
