function results = download_weights()
%DOWNLOAD_WEIGHTS Download the canonical model assets required by the project.
%
% The project expects a single canonical layout under the repository root:
%   models/speecht5/*.onnx
%   models/xvector/*.onnx
%
% This script intentionally downloads only missing assets and leaves existing
% files untouched after a basic integrity check.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
modelRoot = fullfile(repoRoot, 'models');
requiredDirs = {fullfile(modelRoot, 'speecht5'), fullfile(modelRoot, 'xvector')};
for i = 1:numel(requiredDirs)
    if ~isfolder(requiredDirs{i})
        mkdir(requiredDirs{i});
    end
end

assets = struct();
assets(1).name = 'encoder_model.onnx';
assets(1).url = 'https://huggingface.co/microsoft/speecht5_tts/resolve/main/encoder_model.onnx';
assets(1).path = fullfile(modelRoot, 'speecht5', 'encoder_model.onnx');
assets(1).source = 'Microsoft SpeechT5 TTS on Hugging Face';
assets(1).license = 'Microsoft model license; review the source repository before redistribution.';

assets(2).name = 'decoder_model.onnx';
assets(2).url = 'https://huggingface.co/microsoft/speecht5_tts/resolve/main/decoder_model.onnx';
assets(2).path = fullfile(modelRoot, 'speecht5', 'decoder_model.onnx');
assets(2).source = 'Microsoft SpeechT5 TTS on Hugging Face';
assets(2).license = 'Microsoft model license; review the source repository before redistribution.';

assets(3).name = 'decoder_with_past_model.onnx';
assets(3).url = 'https://huggingface.co/microsoft/speecht5_tts/resolve/main/decoder_with_past_model.onnx';
assets(3).path = fullfile(modelRoot, 'speecht5', 'decoder_with_past_model.onnx');
assets(3).source = 'Microsoft SpeechT5 TTS on Hugging Face';
assets(3).license = 'Microsoft model license; review the source repository before redistribution.';

assets(4).name = 'vocoder_model.onnx';
assets(4).url = 'https://huggingface.co/microsoft/speecht5_hifigan/resolve/main/vocoder_model.onnx';
assets(4).path = fullfile(modelRoot, 'speecht5', 'vocoder_model.onnx');
assets(4).source = 'Microsoft SpeechT5 HiFi-GAN vocoder on Hugging Face';
assets(4).license = 'Microsoft model license; review the source repository before redistribution.';

assets(5).name = 'xvector_encoder.onnx';
assets(5).url = 'https://huggingface.co/speechbrain/spkrec-xvect-voxceleb/resolve/main/xvector_encoder.onnx';
assets(5).path = fullfile(modelRoot, 'xvector', 'xvector_encoder.onnx');
assets(5).source = 'SpeechBrain x-vector encoder export';
assets(5).license = 'SpeechBrain project license; review upstream terms before redistribution.';

assets(6).name = 'cmu_arctic_xvectors.mat';
assets(6).url = 'https://huggingface.co/speechbrain/spkrec-xvect-voxceleb/resolve/main/cmu_arctic_xvectors.mat';
assets(6).path = fullfile(modelRoot, 'xvector', 'cmu_arctic_xvectors.mat');
assets(6).source = 'SpeechBrain CMU Arctic lookup table';
assets(6).license = 'CMU Arctic data terms; review the owning dataset license before redistribution.';

results = struct('downloaded', {}, 'skipped', {}, 'failed', {});
for i = 1:numel(assets)
    if isfile(assets(i).path)
        fprintf('[download_weights] Present already: %s\n', assets(i).path);
        results(end + 1).skipped = assets(i).path;
        continue;
    end

    try
        fprintf('[download_weights] Downloading %s -> %s\n', assets(i).name, assets(i).path);
        websave(assets(i).path, assets(i).url);
        if ~isfile(assets(i).path)
            error('download_weights:NoFile', 'Download did not create the expected file.');
        end
        fprintf('[download_weights] OK: %s\n', assets(i).path);
        results(end + 1).downloaded = assets(i).path;
    catch ME
        fprintf('[download_weights] FAILED: %s\n', assets(i).name);
        fprintf('  Source: %s\n', assets(i).url);
        fprintf('  Destination: %s\n', assets(i).path);
        fprintf('  Reason: %s\n', ME.message);
        results(end + 1).failed = struct('asset', assets(i).name, 'url', assets(i).url, 'path', assets(i).path, 'error', ME.message);
    end
end

if any(~cellfun(@isempty, struct2cell(results)))
    fprintf('\n[download_weights] Model download summary complete.\n');
end
end
