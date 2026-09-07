function target_mel = synthesize_features(target_text, speaker_emb, net)
%SYNTHESIZE_FEATURES Run the imported TTS network on text and speaker data.
%   This adapter keeps model-specific tensor packing in one place. The exact
%   input names and shapes must match the exported ONNX model.

arguments
    target_text (1,1) string
    speaker_emb {mustBeNumeric, mustBeFinite}
    net
end

if strlength(strtrim(target_text)) == 0
    error("synthesize_features:EmptyText", "Target text cannot be empty.");
end

text_tokens = double(char(target_text));
inputs = {reshape(text_tokens, 1, []), reshape(double(speaker_emb), 1, [])};
try
    target_mel = predict(net, inputs);
catch exception
    error("synthesize_features:InferenceFailed", ...
        "ONNX inference failed; verify model inputs and export contract: %s", exception.message);
end
end
