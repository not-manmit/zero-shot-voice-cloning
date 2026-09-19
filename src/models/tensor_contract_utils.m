classdef tensor_contract_utils
%TENSOR_CONTRACT_UTILS Centralized tensor transformation and contract enforcement.
%
%   Updated for MATLAB R2026a verified ONNX contracts:
%     ------------------------------------------------------------------------
%     Component        Semantic Layout   Verified ONNX Order  MATLAB Format
%     ------------------------------------------------------------------------
%     Encoder in       [1, T]            [1, T]               'UU'
%     Encoder hidden   [1, T, 768]       [1, T, 768]          'UUU'
%     Decoder Mel in   [1, 1, 80]        [1, 1, 80]           'UUU'
%     Decoder Spk in   [1, 512]          [1, 512]             'UU'
%     Decoder Mask in  [1, T]            [1, T]               'UU'
%     Decoder KV in    [1, 12, seq, 64]  [1, 12, seq, 64]     'UUUU'
%     Decoder Mel out  [rf, 80]          spectrumOutput       Matrix [rf, 80]
%     Decoder Stop out [1, rf]           probOutput           Row [1, rf]
%     Vocoder Mel in   [T_mel, 80]       [T_mel, 80]          'UU'
%     Vocoder Wave out [1, 1, N]         waveformOutput       Vector [N, 1]
%     CAM++ Fbank in   [1, T, 80]        [1, T, 80] ('feats') 'UUU'
%     CAM++ Vector out [1, 512]          embsOutput           Vector [1, 512]
%     ------------------------------------------------------------------------

    methods (Static)
        function [formatted_inputs, in_names] = format_encoder_inputs(varargin)
            %FORMAT_ENCODER_INPUTS Prepare SpeechT5 encoder inputs.
            % Verified contract: input_ids [1, T] format 'UU' int64.
            % (No attention_mask input in verified ONNX model)
            if nargin == 2
                input_ids = varargin{1};
                net = varargin{2};
            elseif nargin >= 3
                input_ids = varargin{1};
                net = varargin{3};
            else
                error("tensor_contract_utils:format_encoder_inputs:InvalidArgs", ...
                    "Requires input_ids and net.");
            end

            in_names = cellstr(net.InputNames);
            input_ids = int64(input_ids(:)');
            T = numel(input_ids);

            if numel(in_names) == 1
                formatted_inputs = dlarray(input_ids, 'UU');
            else
                % Resilient handling if an older split also requests attention_mask
                formatted_inputs = cell(1, numel(in_names));
                for i = 1:numel(in_names)
                    nm = lower(strtrim(in_names{i}));
                    if contains(nm, "input_ids")
                        formatted_inputs{i} = dlarray(input_ids, 'UU');
                    elseif contains(nm, "mask")
                        formatted_inputs{i} = dlarray(ones(1, T, 'int64'), 'UU');
                    else
                        formatted_inputs{i} = dlarray(input_ids, 'UU');
                    end
                end
            end
        end

        function hidden_states = parse_encoder_outputs(raw_out, ~)
            %PARSE_ENCODER_OUTPUTS Extract last_hidden_state [1, T, 768]
            if iscell(raw_out)
                raw = extractdata(raw_out{1});
            else
                raw = extractdata(raw_out);
            end

            raw = single(raw);
            sz = size(raw);

            if ndims(raw) == 2 && sz(2) == 768
                hidden_states = reshape(raw, 1, sz(1), 768);
            elseif ndims(raw) == 3 && sz(3) == 768
                hidden_states = raw;
            elseif ndims(raw) == 3 && sz(2) == 768
                hidden_states = permute(raw, [1, 3, 2]);
            else
                hidden_states = raw;
            end

            if ~all(isfinite(hidden_states), "all")
                error("tensor_contract_utils:NonFiniteEncoderOutput", ...
                    "Encoder output contains non-finite values (NaN or Inf).");
            end
        end

        function [formatted_inputs, in_names] = format_decoder_inputs( ...
                output_sequence, enc_hidden, enc_mask, spk_vec, kv_cache, net, use_past)
            %FORMAT_DECODER_INPUTS Prepare SpeechT5 decoder inputs with explicit 'UU', 'UUU', 'UUUU' formats.
            in_names = cellstr(net.InputNames);
            formatted_inputs = cell(1, numel(in_names));

            for i = 1:numel(in_names)
                nm = in_names{i};
                nm_lower = lower(strtrim(nm));

                if contains(nm_lower, "speaker")
                    % speaker_embeddings: [1, 512] format 'UU'
                    spk_row = single(reshape(spk_vec, 1, 512));
                    formatted_inputs{i} = dlarray(spk_row, 'UU');

                elseif contains(nm_lower, "encoder_hidden") || (contains(nm_lower, "hidden") && ~contains(nm_lower, "mask"))
                    % encoder_hidden_state: [1, T, 768] format 'UUU'
                    enc_h = single(enc_hidden);
                    if ndims(enc_h) == 2
                        enc_h = reshape(enc_h, 1, size(enc_h, 1), 768);
                    end
                    formatted_inputs{i} = dlarray(enc_h, 'UUU');

                elseif contains(nm_lower, "output_sequence")
                    % output_sequence: [1, seq_len, 80] format 'UUU'
                    out_seq = single(output_sequence);
                    if ndims(out_seq) == 2 && size(out_seq, 2) == 80
                        out_seq = reshape(out_seq, 1, size(out_seq, 1), 80);
                    elseif ndims(out_seq) == 2 && size(out_seq, 1) == 80
                        out_seq = reshape(out_seq', 1, size(out_seq, 2), 80);
                    end
                    formatted_inputs{i} = dlarray(out_seq, 'UUU');

                elseif contains(nm_lower, "mask")
                    % encoder_attention_mask: [1, T] format 'UU'
                    mask_row = int64(reshape(enc_mask, 1, []));
                    formatted_inputs{i} = dlarray(mask_row, 'UU');

                elseif contains(nm_lower, "past")
                    % Deterministic KV matching by tensor name
                    tok = regexp(nm, 'past_key_values_(\d+)_(decoder|encoder)_(key|value)', 'tokens', 'once');
                    matched = false;
                    if ~isempty(tok)
                        tag = sprintf('past_key_values_%s_%s_%s', tok{1}, tok{2}, tok{3});
                        if isfield(kv_cache, tag)
                            t_val = kv_cache.(tag);
                            if isempty(dims(t_val)) || dims(t_val) ~= "UUUU"
                                t_val = dlarray(extractdata(t_val), 'UUUU');
                            end
                            formatted_inputs{i} = t_val;
                            matched = true;
                        end
                    end

                    if ~matched
                        if isfield(kv_cache, nm)
                            t_val = kv_cache.(nm);
                            formatted_inputs{i} = dlarray(extractdata(t_val), 'UUUU');
                        else
                            if use_past
                                error("tensor_contract_utils:MissingPastKVTensor", ...
                                    "Decoder input '%s' requires cached KV tensor, but tensor not found in kv_cache.", nm);
                            else
                                % Placeholder tensor for first step if model requests it
                                formatted_inputs{i} = dlarray(single(zeros(1, 12, 1, 64)), 'UUUU');
                            end
                        end
                    end

                elseif contains(nm_lower, "use_cache")
                    % Merged model flag
                    formatted_inputs{i} = dlarray(logical(use_past), 'UU');

                else
                    error("tensor_contract_utils:UnrecognizedDecoderInput", ...
                        "Unrecognized decoder input '%s' at position %d.", nm, i);
                end
            end

            if numel(formatted_inputs) == 1
                formatted_inputs = formatted_inputs{1};
            end
        end

        function kv_cache = update_kv_cache(kv_cache, raw_out, net)
            %UPDATE_KV_CACHE Deterministically map present KV outputs to past KV inputs.
            %
            % Mapping Contract:
            %   present_L_decoder_key   -> past_key_values_L_decoder_key
            %   present_L_decoder_value -> past_key_values_L_decoder_value
            %   present_L_encoder_key   -> past_key_values_L_encoder_key
            %   present_L_encoder_value -> past_key_values_L_encoder_value
            %
            % Note: First-step decoder emits 24 tensors (12 decoder + 12 encoder).
            %       decoder_with_past emits 12 updated decoder tensors only.
            %       Encoder KV tensors are retained across all subsequent steps.

            if isempty(kv_cache)
                kv_cache = struct();
            end

            out_names = cellstr(net.OutputNames);

            if iscell(raw_out)
                for k = 1:numel(raw_out)
                    if k > numel(out_names)
                        continue;
                    end
                    nm = out_names{k};
                    tok = regexp(nm, 'present_(\d+)_(decoder|encoder)_(key|value)', 'tokens', 'once');
                    if ~isempty(tok)
                        dest_tag = sprintf('past_key_values_%s_%s_%s', tok{1}, tok{2}, tok{3});
                        val = extractdata(raw_out{k});
                        kv_cache.(dest_tag) = dlarray(single(val), 'UUUU');
                    end
                end
            end
        end

        function [spectrum, prob_val, raw_logit, kv_cache] = parse_decoder_outputs(raw_out, net, rf, kv_cache)
            %PARSE_DECODER_OUTPUTS Extract spectrumOutput, probOutput, and update KV cache.
            if nargin < 4
                kv_cache = struct();
            end

            out_names = cellstr(net.OutputNames);
            spectrum_raw = [];
            prob_raw = [];

            if iscell(raw_out)
                for k = 1:numel(raw_out)
                    if k > numel(out_names)
                        continue;
                    end
                    nm = out_names{k};

                    if strcmp(nm, "spectrumOutput") || (contains(nm, "spectrum", "IgnoreCase", true) && isempty(spectrum_raw))
                        spectrum_raw = extractdata(raw_out{k});
                    elseif strcmp(nm, "probOutput") || (contains(nm, "prob", "IgnoreCase", true) && isempty(prob_raw))
                        prob_raw = extractdata(raw_out{k});
                    end
                end

                % Update KV cache deterministically by tensor name
                kv_cache = tensor_contract_utils.update_kv_cache(kv_cache, raw_out, net);
            else
                spectrum_raw = extractdata(raw_out);
            end

            % Fallback if exact names were absent
            if isempty(spectrum_raw) && iscell(raw_out) && numel(raw_out) >= 1
                spectrum_raw = extractdata(raw_out{1});
            end
            if isempty(prob_raw) && iscell(raw_out) && numel(raw_out) >= 2
                cand = extractdata(raw_out{2});
                if numel(cand) <= 8
                    prob_raw = cand;
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
                raw = repmat(raw(1), 1, rf);
            end

            % If values are outside [0, 1], apply logistic sigmoid
            p = raw;
            is_logit = any(p < 0 | p > 1);
            if is_logit
                p = 1.0 ./ (1.0 + exp(-double(p)));
                p = single(p);
            end
        end

        function [formatted_input, in_name] = format_campp_inputs(fbank80, net)
            %FORMAT_CAMPP_INPUTS Format 80-bin filterbank for CAM++ speaker encoder.
            % Exact contract: 'feats' [1, T, 80] format 'UUU'.
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

            % Shape = [1, T, 80], Format = 'UUU'
            batched = reshape(fbank80, 1, size(fbank80, 1), 80);
            formatted_input = dlarray(batched, 'UUU');
        end

        function [formatted_input, in_name] = format_vocoder_inputs(mel80, net)
            %FORMAT_VOCODER_INPUTS Format 80-bin log-Mel for HiFi-GAN vocoder.
            % Exact contract: 'spectrogram' [T_mel, 80] format 'UU'.
            % (The ONNX model internally performs required transpose/reshape).
            in_names = cellstr(net.InputNames);
            in_name = in_names{1};

            mel = single(mel80);
            if size(mel, 1) == 80
                % [80, T_mel] -> transpose to raw ONNX order [T_mel, 80]
                mel = mel';
            end
            if size(mel, 2) ~= 80
                error("tensor_contract_utils:VocoderBadMelShape", ...
                    "Vocoder requires 80 Mel bins in column 2, got size %s", mat2str(size(mel)));
            end

            % Shape = [T_mel, 80], Format = 'UU'
            formatted_input = dlarray(mel, 'UU');
        end
    end
end
