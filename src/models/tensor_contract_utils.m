classdef tensor_contract_utils
%TENSOR_CONTRACT_UTILS Centralized tensor transformation and contract enforcement.
%
%   All dlarray conversions, dimension permutations, shape validations,
%   and layout interpretations are centralized in this class.
%
%   ONNX Tensor Convention vs MATLAB dlarray Layouts:
%     ------------------------------------------------------------------------
%     Component        Semantic Layout   ONNX Format    MATLAB dlarray Format
%     ------------------------------------------------------------------------
%     Encoder inputs   [Batch, Seq]      [B, T]         'CB' (Channel=Seq, Batch=B)
%     Encoder hidden   [Batch, Seq, H]   [B, T, 768]    'CB' or [1, T, 768]
%     Decoder Mel in   [Batch, Seq, Mel] [1, 1, 80]     'CBT' (Mel=C, Batch=B, Time=T)
%     Decoder Mel out  [Batch, rf, Mel]  [1, 2, 80]     Matrix [rf, 80]
%     Decoder Stop out [Batch, rf]       [1, 2]         Vector [1, rf] (logits or prob)
%     Vocoder Mel in   [Batch, Mel, T]   [1, 80, T]     'SCB' or [1, 80, T]
%     Vocoder Wave out [Batch, 1, N]     [1, 1, N]      Vector [N, 1]
%     CAM++ Fbank in   [Batch, T, Mel]   [1, T, 80]     'CBT'
%     CAM++ Vector out [Batch, 512]      [1, 512]       Vector [1, 512]
%     ------------------------------------------------------------------------

    methods (Static)
        function [formatted_inputs, in_names] = format_encoder_inputs(input_ids, attention_mask, net)
            %FORMAT_ENCODER_INPUTS Prepare SpeechT5 encoder inputs
            % Expected: int64 input_ids [1, T], int64 attention_mask [1, T]
            in_names = cellstr(net.InputNames);
            if numel(in_names) ~= 2
                error("tensor_contract_utils:EncoderInputCountMismatch", ...
                    "Encoder expected 2 inputs (input_ids, attention_mask), observed %d: [%s]", ...
                    numel(in_names), strjoin(in_names, ", "));
            end

            input_ids = int64(input_ids(:)');
            attention_mask = int64(attention_mask(:)');
            T = numel(input_ids);

            if numel(attention_mask) ~= T
                error("tensor_contract_utils:EncoderLengthMismatch", ...
                    "input_ids length (%d) does not match attention_mask length (%d)", T, numel(attention_mask));
            end

            formatted_inputs = cell(1, numel(in_names));
            for i = 1:numel(in_names)
                name_lower = lower(strtrim(in_names{i}));
                if contains(name_lower, "input_ids")
                    formatted_inputs{i} = dlarray(input_ids, 'CB');
                elseif contains(name_lower, "mask")
                    formatted_inputs{i} = dlarray(attention_mask, 'CB');
                else
                    error("tensor_contract_utils:UnexpectedEncoderInput", ...
                        "Unexpected encoder input name '%s' at index %d. Expected input_ids or attention_mask.", ...
                        in_names{i}, i);
                end
            end

            if numel(formatted_inputs) == 1
                formatted_inputs = formatted_inputs{1};
            end
        end

        function hidden_states = parse_encoder_outputs(raw_out, net)
            %PARSE_ENCODER_OUTPUTS Extract last_hidden_state [1, T, 768]
            if iscell(raw_out)
                raw = extractdata(raw_out{1});
            else
                raw = extractdata(raw_out);
            end

            raw = single(raw);
            sz = size(raw);

            % Expected shape is [1, T, 768] or [T, 768]
            if ndims(raw) == 2 && sz(2) == 768
                hidden_states = reshape(raw, 1, sz(1), sz(2));
            elseif ndims(raw) == 3 && sz(3) == 768
                hidden_states = raw;
            elseif ndims(raw) == 3 && sz(2) == 768
                % Permute if [B, 768, T]
                hidden_states = permute(raw, [1, 3, 2]);
            else
                hidden_states = raw;
                warning("tensor_contract_utils:EncoderHiddenDimWarning", ...
                    "Encoder hidden state dimensions %s differ from canonical [1, T, 768]", mat2str(sz));
            end

            if ~all(isfinite(hidden_states), "all")
                error("tensor_contract_utils:NonFiniteEncoderOutput", ...
                    "Encoder output contains non-finite values (NaN or Inf).");
            end
        end

        function [formatted_inputs, in_names] = format_decoder_inputs(output_sequence, enc_hidden, enc_mask, spk_vec, past_kv, net, use_past)
            %FORMAT_DECODER_INPUTS Prepare SpeechT5 decoder inputs (both first-step and with-past)
            in_names = cellstr(net.InputNames);
            formatted_inputs = cell(1, numel(in_names));
            past_idx = 1;

            has_use_cache = any(contains(lower(in_names), "use_cache"));
            is_float_mel = any(contains(lower(in_names), "output_sequence")) || ...
                           any(contains(lower(in_names), "input_values")) || ...
                           any(contains(lower(in_names), "spectrogram"));
            is_legacy_int = ~is_float_mel && any(contains(lower(in_names), "input_ids"));

            for i = 1:numel(in_names)
                nm = lower(strtrim(in_names{i}));

                if has_use_cache && contains(nm, "use_cache")
                    % Merged export branch flag: bool true when past is present
                    formatted_inputs{i} = dlarray(logical(use_past), 'CB');
                elseif contains(nm, "output_sequence") || contains(nm, "input_values") || contains(nm, "spectrogram")
                    % Float Mel frame [1, 1, 80]
                    mel_frame = single(reshape(output_sequence, 1, 1, 80));
                    try
                        formatted_inputs{i} = dlarray(mel_frame, 'CBT');
                    catch
                        formatted_inputs{i} = dlarray(mel_frame, 'SCB');
                    end
                elseif contains(nm, "input_ids") && is_legacy_int
                    % Legacy BOS int64 fallback
                    formatted_inputs{i} = dlarray(int64(0), 'CB');
                elseif contains(nm, "encoder_hidden_states") || (contains(nm, "hidden") && ~contains(nm, "mask"))
                    formatted_inputs{i} = dlarray(single(enc_hidden), 'CB');
                elseif contains(nm, "encoder_attention_mask") || (contains(nm, "mask") && contains(nm, "encoder"))
                    formatted_inputs{i} = dlarray(int64(enc_mask), 'CB');
                elseif contains(nm, "speaker_embeddings") || contains(nm, "speaker_embedding")
                    spk_row = single(reshape(spk_vec, 1, 512));
                    formatted_inputs{i} = dlarray(spk_row, 'CB');
                elseif contains(nm, "past") || contains(nm, "cache") || contains(nm, "present") || ...
                       contains(nm, "key") || contains(nm, "value") || contains(nm, "pkv")
                    if ~isempty(past_kv) && past_idx <= numel(past_kv)
                        formatted_inputs{i} = past_kv{past_idx};
                        past_idx = past_idx + 1;
                    else
                        if use_past
                            error("tensor_contract_utils:MissingPastKVTensor", ...
                                "Decoder input '%s' (input #%d) requires cached KV tensor, but pastKV cache has only %d entries.", ...
                                in_names{i}, i, numel(past_kv));
                        else
                            % Dummy empty tensor if first step expects placeholder past input
                            formatted_inputs{i} = dlarray(single(zeros(1, 1, 1, 1)), 'CB');
                        end
                    end
                else
                    error("tensor_contract_utils:UnrecognizedDecoderInput", ...
                        "Unrecognized decoder input '%s' (index %d). Inspect with scripts/diagnose_onnx.m.", ...
                        in_names{i}, i);
                end
            end

            if numel(formatted_inputs) == 1
                formatted_inputs = formatted_inputs{1};
            end
        end

        function [spectrum, prob_val, raw_logit, new_past_kv] = parse_decoder_outputs(raw_out, net, rf)
            %PARSE_DECODER_OUTPUTS Deconstruct decoder outputs into Mel spectrum, stop probability, and updated KV cache.
            %
            % Outputs:
            %   spectrum    – [rf, 80] float32 predicted Mel frames
            %   prob_val    – [1, rf] float32 sigmoid stop probabilities in [0, 1]
            %   raw_logit   – [1, rf] float32 un-sigmoid logit values
            %   new_past_kv – cell array of dlarrays containing KV cache tensors

            out_names = {};
            if isprop(net, 'OutputNames')
                out_names = lower(cellstr(net.OutputNames));
            end

            spectrum_raw = [];
            prob_raw = [];
            new_past_kv = {};

            if iscell(raw_out)
                n_out = numel(raw_out);
                for k = 1:n_out
                    nm = "";
                    if k <= numel(out_names)
                        nm = out_names{k};
                    end

                    if (contains(nm, "prob") || contains(nm, "logit")) && isempty(prob_raw)
                        prob_raw = extractdata(raw_out{k});
                    elseif (contains(nm, "spectrum") || contains(nm, "feat") || contains(nm, "mel") || strcmp(nm, "logits")) && isempty(spectrum_raw)
                        spectrum_raw = extractdata(raw_out{k});
                    elseif contains(nm, "past") || contains(nm, "present") || contains(nm, "key") || contains(nm, "value")
                        new_past_kv{end+1} = raw_out{k}; %#ok<AGROW>
                    else
                        % Name-less or unclassified fallback by order
                        if isempty(spectrum_raw)
                            spectrum_raw = extractdata(raw_out{k});
                        elseif isempty(prob_raw) && k == 2
                            cand = extractdata(raw_out{k});
                            if numel(cand) <= 8
                                prob_raw = cand;
                            else
                                new_past_kv{end+1} = raw_out{k}; %#ok<AGROW>
                            end
                        else
                            new_past_kv{end+1} = raw_out{k}; %#ok<AGROW>
                        end
                    end
                end
            else
                spectrum_raw = extractdata(raw_out);
            end

            % Handle legacy packed 81-dim output (80 Mel bins + 1 stop logit)
            if ~isempty(spectrum_raw) && isempty(prob_raw)
                sz = size(spectrum_raw);
                if sz(end) == 81
                    if ndims(spectrum_raw) == 3
                        prob_raw = squeeze(spectrum_raw(:, :, 81));
                        spectrum_raw = spectrum_raw(:, :, 1:80);
                    elseif ndims(spectrum_raw) == 2
                        prob_raw = spectrum_raw(end);
                        spectrum_raw = spectrum_raw(1:80);
                    end
                end
            end

            % Normalize spectrum to [rf, 80]
            spectrum = tensor_contract_utils.normalize_spectrum(spectrum_raw, rf);

            % Normalize stop probability
            [prob_val, raw_logit] = tensor_contract_utils.normalize_probability(prob_raw, rf);
        end

        function s = normalize_spectrum(raw, rf)
            %NORMALIZE_SPECTRUM Standardize predicted Mel to [rf, 80] frames
            if isempty(raw)
                s = single(zeros(0, 80));
                return;
            end
            raw = single(raw);
            sz = size(raw);

            if ndims(raw) == 3
                if sz(3) == 80
                    s = reshape(raw, [], 80);
                elseif sz(2) == 80
                    s = permute(raw, [1, 3, 2]);
                    s = reshape(s, [], 80);
                else
                    s = reshape(raw, [], 80);
                end
            elseif ndims(raw) == 2
                if sz(2) == 80
                    s = raw;
                elseif sz(1) == 80
                    s = raw';
                else
                    s = reshape(raw, [], 80);
                end
            else
                s = reshape(raw, [], 80);
            end

            if size(s, 2) ~= 80 && size(s, 1) == 80
                s = s';
            end
        end

        function [p, raw] = normalize_probability(raw_in, rf)
            %NORMALIZE_PROBABILITY Compute calibrated [0, 1] probability from raw logits
            if isempty(raw_in)
                p = single(zeros(1, rf));
                raw = single(zeros(1, rf));
                return;
            end

            raw = single(raw_in(:)');
            if numel(raw) > rf
                raw = raw(1:rf);
            elseif numel(raw) < rf && numel(raw) > 0
                % Expand if scalar stop for rf frames
                raw = repmat(raw(1), 1, rf);
            end

            % Distinguish raw logit vs probability:
            % If any value is outside [0, 1], it is a raw logit and must pass through sigmoid.
            p = raw;
            is_logit = any(p < 0 | p > 1);
            if is_logit
                p = 1.0 ./ (1.0 + exp(-double(p)));
                p = single(p);
            end
        end

        function [formatted_input, in_name] = format_campp_inputs(fbank80, net)
            %FORMAT_CAMPP_INPUTS Format 80-bin filterbank for CAM++ speaker encoder
            % Expected: fbank80 [T, 80], layout CBT [1, T, 80]
            in_names = cellstr(net.InputNames);
            in_name = in_names{1};

            fbank80 = single(fbank80);
            if size(fbank80, 2) ~= 80 && size(fbank80, 1) == 80
                fbank80 = fbank80';
            end
            if size(fbank80, 2) ~= 80
                error("tensor_contract_utils:CAMppBadDim", ...
                    "CAM++ fbank must have 80 columns, got size %s", mat2str(size(fbank80)));
            end

            batched = reshape(fbank80, 1, size(fbank80, 1), 80); % [1, T, 80]
            formatted_input = dlarray(batched, 'CBT');
        end

        function [formatted_input, in_name] = format_vocoder_inputs(mel80, net)
            %FORMAT_VOCODER_INPUTS Format 80-bin log-Mel for HiFi-GAN vocoder
            % Expected: mel80 [80, T] -> [1, 80, T] SCB
            in_names = cellstr(net.InputNames);
            in_name = in_names{1};

            mel = single(mel80);
            if size(mel, 1) ~= 80 && size(mel, 2) == 80
                mel = mel';
            end
            if size(mel, 1) ~= 80
                error("tensor_contract_utils:VocoderBadMelShape", ...
                    "Vocoder requires 80 Mel bins in row 1, got size %s", mat2str(size(mel)));
            end

            batched = reshape(mel, 1, 80, size(mel, 2)); % [1, 80, T]
            formatted_input = dlarray(batched, 'SCB');
        end
    end
end
