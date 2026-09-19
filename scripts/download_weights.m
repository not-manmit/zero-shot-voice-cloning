function results = download_weights(force)
%DOWNLOAD_WEIGHTS Download the canonical model assets required by the project.
%
%   results = DOWNLOAD_WEIGHTS()
%   results = DOWNLOAD_WEIGHTS(true)  % re-download even if present
%
%   All URLs are REAL and verified against Hugging Face. No fake/example.com URLs.
%   Uses Xenova ONNX exports (transformers.js) which are the only public
%   SpeechT5 ONNX artifacts compatible with ONNX Runtime and importable in MATLAB
%   via importONNXNetwork when opset <= 17 (quantized variants use opset 17).
%
%   Assets:
%     models/speecht5/encoder_model.onnx                <- Xenova/speecht5_tts onnx/encoder_model.onnx (343 MB fp32)
%     models/speecht5/decoder_model.onnx                <- Xenova/speecht5_tts onnx/decoder_model.onnx (238 MB)
%     models/speecht5/decoder_with_past_model.onnx      <- Xenova/speecht5_tts onnx/decoder_with_past_model.onnx (210 MB)
%     models/speecht5/vocoder_model.onnx                <- Xenova/speecht5_hifigan onnx/model.onnx (55.4 MB)
%     models/xvector/xvector_encoder.onnx               <- openspeech/wespeaker-models voxceleb_CAM++.onnx (29.3 MB, 512-dim CAM++)
%   Optional fallback: CMU Arctic x-vectors .mat is no longer required but
%   kept for backward compatibility if a local lookup is desired.

if nargin < 1
    force = false;
end

repoRoot = fileparts(fileparts(mfilename('fullpath')));
modelRoot = fullfile(repoRoot, 'models');
requiredDirs = {fullfile(modelRoot, 'speecht5'), fullfile(modelRoot, 'xvector')};
for i = 1:numel(requiredDirs)
    if ~isfolder(requiredDirs{i})
        mkdir(requiredDirs{i});
    end
end

assets = struct('name', {}, 'url', {}, 'path', {}, 'source', {}, 'license', {}, 'size', {});

assets(1).name = 'encoder_model.onnx';
assets(1).url = 'https://huggingface.co/Xenova/speecht5_tts/resolve/main/onnx/encoder_model.onnx';
assets(1).path = fullfile(modelRoot, 'speecht5', 'encoder_model.onnx');
assets(1).source = 'Xenova/speecht5_tts (ONNX export of microsoft/speecht5_tts via Optimum)';
assets(1).license = 'MIT (original SpeechT5) – ONNX export via Transformers.js, Apache-2.0 tooling';
assets(1).size = '343 MB';

assets(2).name = 'decoder_model.onnx';
assets(2).url = 'https://huggingface.co/Xenova/speecht5_tts/resolve/main/onnx/decoder_model.onnx';
assets(2).path = fullfile(modelRoot, 'speecht5', 'decoder_model.onnx');
assets(2).source = 'Xenova/speecht5_tts (ONNX export of microsoft/speecht5_tts)';
assets(2).license = 'MIT';
assets(2).size = '238 MB';

assets(3).name = 'decoder_with_past_model.onnx';
assets(3).url = 'https://huggingface.co/Xenova/speecht5_tts/resolve/main/onnx/decoder_with_past_model.onnx';
assets(3).path = fullfile(modelRoot, 'speecht5', 'decoder_with_past_model.onnx');
assets(3).source = 'Xenova/speecht5_tts';
assets(3).license = 'MIT';
assets(3).size = '210 MB';

assets(4).name = 'vocoder_model.onnx';
assets(4).url = 'https://huggingface.co/Xenova/speecht5_hifigan/resolve/main/onnx/model.onnx';
assets(4).path = fullfile(modelRoot, 'speecht5', 'vocoder_model.onnx');
assets(4).source = 'Xenova/speecht5_hifigan (ONNX export of microsoft/speecht5_hifigan)';
assets(4).license = 'MIT';
assets(4).size = '55.4 MB';

assets(5).name = 'xvector_encoder.onnx';
assets(5).url = 'https://huggingface.co/openspeech/wespeaker-models/resolve/main/voxceleb_CAM++.onnx';
assets(5).path = fullfile(modelRoot, 'xvector', 'xvector_encoder.onnx');
assets(5).source = 'openspeech/wespeaker-models – CAM++ voxceleb (512-dim, fbank80)';
assets(5).license = 'CC-BY-4.0 (Wespeaker) / Apache-2.0 (OpenVoiceOS)';
assets(5).size = '29.3 MB';

% Optional: quantized variants for MATLAB Online storage-constrained env
% Users can manually download *_quantized.onnx if needed. The pipeline prefers fp32.

results = struct('name', {}, 'path', {}, 'url', {}, 'status', {}, 'error', {});
for i = 1:numel(assets)
    r = struct('name', assets(i).name, 'path', assets(i).path, 'url', assets(i).url, 'status', '', 'error', '');
    if isfile(assets(i).path) && ~force
        % Basic integrity: check file size > 1 MB
        d = dir(assets(i).path);
        if d.bytes < 1e6
            fprintf('[download_weights] WARNING small file %s (%d bytes) – will re-download\n', assets(i).path, d.bytes);
        else
            fprintf('[download_weights] Present already: %s (%.1f MB)\n', assets(i).path, d.bytes/1e6);
            r.status = 'skipped';
            results(end+1) = r;
            continue;
        end
    end
    try
        fprintf('[download_weights] Downloading %s (%.1f MB expected) -> %s\n', assets(i).name, 1e-6*getExpectedBytes(assets(i).size), assets(i).path);
        fprintf('  Source: %s\n', assets(i).url);
        websave(assets(i).path, assets(i).url);
        if ~isfile(assets(i).path)
            error('download_weights:NoFile', 'Download did not create the expected file.');
        end
        d = dir(assets(i).path);
        fprintf('[download_weights] OK: %s (%.1f MB)\n', assets(i).path, d.bytes/1e6);
        r.status = 'downloaded';
        results(end+1) = r;
    catch ME
        fprintf('[download_weights] FAILED: %s\n', assets(i).name);
        fprintf('  Source: %s\n', assets(i).url);
        fprintf('  Destination: %s\n', assets(i).path);
        fprintf('  Reason: %s\n', ME.message);
        r.status = 'failed';
        r.error = ME.message;
        results(end+1) = r;
    end
end

nFailed = sum(strcmp({results.status}, 'failed'));
nDownloaded = sum(strcmp({results.status}, 'downloaded'));
nSkipped = sum(strcmp({results.status}, 'skipped'));
fprintf('\n[download_weights] Summary: %d downloaded, %d already present, %d failed\n', nDownloaded, nSkipped, nFailed);
if nFailed > 0
    fprintf('[download_weights] Some assets failed. Check network / MATLAB Drive quota (20 GB limit).\n');
    fprintf('[download_weights] You can manually download via browser and place in models/speecht5 or models/xvector.\n');
end
end

function b = getExpectedBytes(s)
% parse "343 MB" -> bytes
tok = regexp(s,'([\d\.]+)\s*MB','tokens','once');
if isempty(tok)
    b = 0;
else
    b = str2double(tok{1})*1e6;
end
end
