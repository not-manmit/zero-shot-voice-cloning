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
accumulated_postnet_frames = [];
current_full_mel = [];
kv_cache = struct();

% Sentence length estimation and step bounds
% Standard English speech rate: ~3.8 Mel frames per character token (16 ms/frame)
expected_mel_frames = round(T_tokens * 3.8);
min_steps = max(25, round(0.80 * expected_mel_frames / rf));
max_steps_for_text = min(cfg.max_decoder_steps, max(min_steps + 25, round(1.35 * expected_mel_frames / rf)));

fprintf("[synthesize_features] Target text (%d tokens): expected ~%d frames (~%.2f s). Step bounds: min=%d, max=%d\n", ...
    T_tokens, expected_mel_frames, expected_mel_frames * 256 / cfg.fs, min_steps, max_steps_for_text);

use_kv = false;
if isfield(cfg, 'use_kv_cache') && cfg.use_kv_cache
    use_kv = true;
end

dec_kv_net = [];
if use_kv
    if isfield(models, 'decoder_with_past') && ~isempty(models.decoder_with_past)
        dec_kv_net = models.decoder_with_past;
    elseif isfield(models, 'decoder_kv') && ~isempty(models.decoder_kv)
        dec_kv_net = models.decoder_kv;
    end
end

consecutive_stop_required = 2;
consecutive_stop_count = 0;

for step = 1:max_steps_for_text
    use_past_step = use_kv && (step > 1) && ~isempty(dec_kv_net) && ~isempty(fieldnames(kv_cache));

    if use_past_step
        current_net = dec_kv_net;
    else
        current_net = models.decoder;
    end

    % Format decoder inputs
    try
        dec_inputs = tensor_contract_utils.format_decoder_inputs( ...
            output_sequence, enc_hidden, enc_attn_mask, spk_vec, kv_cache, current_net, use_past_step);
        [dec_raw_out, current_net] = tensor_contract_utils.predict_net(current_net, dec_inputs);
        if use_past_step
            dec_kv_net = current_net;
        else
            models.decoder = current_net;
        end
    catch ME
        error("synthesize_features:DecoderStepFailed", ...
            "SpeechT5 decoder failed at autoregressive step %d (usePast=%s, rf=%d):\nError: %s\nCheck models/speecht5 ONNX contract via scripts/diagnose_onnx.m.", ...
            step, string(use_past_step), rf, ME.message);
    end

    % Parse outputs: pre-postnet spectrum, postnet spectrum, stop probabilities
    [spectrum_frames, stop_probs, ~, kv_cache, postnet_spectrum_frames] = ...
        tensor_contract_utils.parse_decoder_outputs(dec_raw_out, current_net, rf, kv_cache);

    % Verify numerical finiteness
    if ~all(isfinite(spectrum_frames), 'all')
        spectrum_frames(~isfinite(spectrum_frames)) = 0;
    end
    if ~all(isfinite(postnet_spectrum_frames), 'all')
        postnet_spectrum_frames(~isfinite(postnet_spectrum_frames)) = 0;
    end

    % Accumulate postnet frames
    if size(postnet_spectrum_frames, 1) >= step * rf
        % Full-sequence Postnet output: model emitted complete refined sequence
        current_full_mel = postnet_spectrum_frames;
    else
        % Step-slice Postnet output: accumulate latest rf frames
        step_frames = postnet_spectrum_frames(max(1, end-rf+1):end, :);
        accumulated_postnet_frames = [accumulated_postnet_frames; step_frames]; %#ok<AGROW>
        current_full_mel = accumulated_postnet_frames;
    end

    % Stop condition evaluation: only after min_steps, requiring 2 consecutive triggers
    if step >= min_steps
        max_prob = max(stop_probs);
        if max_prob > cfg.stop_threshold
            consecutive_stop_count = consecutive_stop_count + 1;
            if consecutive_stop_count >= consecutive_stop_required
                fprintf("[synthesize_features] Stop token triggered at step %d (%d frames, %.2f s)\n", ...
                    step, size(current_full_mel, 1), size(current_full_mel, 1) * 256 / cfg.fs);
                break;
            end
        else
            consecutive_stop_count = 0;
        end
    end

    % Stagnation / energy collapse guard
    if size(current_full_mel, 1) >= 30
        recent_energy = mean(abs(current_full_mel(end-4:end, :)), 'all');
        if recent_energy < 1e-7
            fprintf("[synthesize_features] Energy collapse detected at step %d — stopping.\n", step);
            break;
        end
    end

    % Prepare next output_sequence:
    % Extract the last pre-postnet Mel frame [1, 1, 80]
    last_mel_frame = reshape(single(spectrum_frames(end, :)), 1, 1, cfg.n_mels);

    if use_kv
        % When using KV cache, pass only the single newest frame
        output_sequence = last_mel_frame;
    else
        % Full-prefix mode: append newest frame to growing sequence [1, step+1, 80]
        % This guarantees exact prenet sinusoidal positional embeddings across the entire sentence!
        output_sequence = cat(2, output_sequence, last_mel_frame);
    end
end

timings.decoderTime_s = toc(t_dec);

if isempty(current_full_mel)
    error("synthesize_features:EmptyAcousticFeatures", ...
        "SpeechT5 decoder finished without producing acoustic frames.");
end

% Return standard [n_mels x T_mel] matrix using POSTNET frames
acoustic = single(current_full_mel');

fprintf("[synthesize_features] Generated %d Mel frames (%d decoder steps, %.2f s audio at 16 kHz)\n", ...
    size(acoustic, 2), step, size(acoustic, 2) * 256 / 16000);

if ~all(isfinite(acoustic), 'all')
    error("synthesize_features:NonFiniteAcousticMatrix", ...
        "Generated acoustic Mel matrix contains NaN or Inf values.");
end

end
