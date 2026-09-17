function spk_emb = extract_speaker_embedding(x, fs, models, cfg)
%EXTRACT_SPEAKER_EMBEDDING  Derive a 512-dim L2-normalised x-vector.
%
%   spk_emb = EXTRACT_SPEAKER_EMBEDDING(x, fs, models)
%   spk_emb = EXTRACT_SPEAKER_EMBEDDING(x, fs, models, cfg)
%
%   Runs the SpeechBrain TDNN x-vector encoder (ONNX) on the reference
%   waveform to produce a 512-dimensional speaker embedding that is
%   subsequently consumed by the SpeechT5 TTS decoder.
%
%   The embedding is L2-normalised so that its Euclidean norm equals 1,
%   matching the convention used during SpeechT5 pre-training.
%
%   For reference clips longer than cfg.spk_chunk_dur_s seconds, the audio
%   is split into overlapping segments, an embedding is computed for each,
%   and the results are averaged before normalisation.  This mirrors the
%   multi-segment averaging described in the SpeechBrain documentation.
%
%   Preprocessing inside this function
%   -----------------------------------
%   1.  Enforce mono, 16 kHz (calls preprocess_signal if needed).
%   2.  Validate duration (≥ cfg.spk_min_dur_s).
%   3.  Validate energy (reject silent / near-silent clips).
%   4.  Chunk into segments if clip is long.
%   5.  Pass each segment through the ONNX x-vector encoder.
%   6.  Average embeddings, L2-normalise.
%
%   Inputs
%   ------
%   x       – mono/stereo waveform column vector (double, [-1,1])
%   fs      – sample rate of x (Hz)
%   models  – struct from load_onnx_engine(); uses models.spk_encoder
%   cfg     – (optional) struct from pipeline_config()
%
%   Output
%   ------
%   spk_emb – float32 row vector [1 × 512],  L2-norm ≈ 1

arguments
    x      (:,:) {mustBeNumeric, mustBeFinite}
    fs     (1,1) {mustBePositive, mustBeFinite}
    models struct
    cfg    struct = struct()
end

% -----------------------------------------------------------------------
% Resolve configuration
% -----------------------------------------------------------------------
if ~isfield(cfg, 'fs')
    cfg = pipeline_config();
end

% -----------------------------------------------------------------------
% Step 1 – Ensure mono, 16 kHz, noise-suppressed, peak-normalised
% -----------------------------------------------------------------------
if ~isvector(x)
    x = mean(x, 2);
end
x = x(:);

if abs(fs - cfg.fs) > 1
    [x, fs] = preprocess_signal(x, fs, cfg.fs);
end

% -----------------------------------------------------------------------
% Step 2 – Duration validation
% -----------------------------------------------------------------------
dur_s = numel(x) / fs;
if dur_s < cfg.spk_min_dur_s
    error("extract_speaker_embedding:TooShort", ...
        "Reference audio is %.2f s; minimum required is %.2f s. " + ...
        "Record or upload a longer clip.", dur_s, cfg.spk_min_dur_s);
end

% -----------------------------------------------------------------------
% Step 3 – Energy / silence check
% -----------------------------------------------------------------------
rms_val = sqrt(mean(x .^ 2));
if rms_val < 1e-4
    error("extract_speaker_embedding:SilentAudio", ...
        "Reference audio appears silent (RMS = %.2e). " + ...
        "Check microphone level or WAV content.", rms_val);
end

% -----------------------------------------------------------------------
% Step 4 – Chunk into segments of cfg.spk_chunk_dur_s with 50 %% overlap
% -----------------------------------------------------------------------
chunk_len = round(cfg.spk_chunk_dur_s * fs);
hop_len   = round(chunk_len / 2);
n_samples = numel(x);

starts = 1 : hop_len : max(1, n_samples - chunk_len + 1);
embeddings = zeros(numel(starts), cfg.spk_emb_dim, "single");

for k = 1:numel(starts)
    s = starts(k);
    e = min(s + chunk_len - 1, n_samples);
    seg = x(s:e);

    % Zero-pad very short final segment to at least 1 s
    min_len = round(cfg.spk_min_dur_s * fs);
    if numel(seg) < min_len
        seg(end+1:min_len) = 0;
    end

    embeddings(k, :) = run_xvector_onnx(seg, fs, models, cfg);
end

% -----------------------------------------------------------------------
% Step 5 – Average and L2-normalise
% -----------------------------------------------------------------------
mean_emb = mean(embeddings, 1);          % [1 × 512]
norm_val  = norm(mean_emb, 2);
if norm_val < eps
    error("extract_speaker_embedding:ZeroEmbedding", ...
        "Speaker embedding has near-zero norm after averaging. " + ...
        "Check x-vector model and reference audio.");
end
spk_emb = single(mean_emb ./ norm_val);  % float32, L2-norm = 1
end


% -----------------------------------------------------------------------
% Local helper – run one ONNX x-vector forward pass
% -----------------------------------------------------------------------
function emb = run_xvector_onnx(x_seg, fs, models, cfg)
%RUN_XVECTOR_ONNX  Single-segment x-vector inference via ONNX.
%
%   The SpeechBrain spkrec-xvect-voxceleb ONNX model expects:
%       Input  : float32  [1 × 1 × T_samples]   raw waveform at 16 kHz
%       Output : float32  [1 × D_emb]            embedding (pre-norm)
%
%   MATLAB's importONNXNetwork flattens the batch dimension, so the
%   effective shapes seen by predict() are:
%       Input  : dlarray  [1 × T_samples]   (format 'CB' or 'CT')
%       Output : dlarray  [1 × D_emb]

% Build input tensor ------------------------------------------------
x_f32 = single(x_seg(:)');                       % row vector [1 × T]
inp   = dlarray(x_f32, 'CB');                    % 'C'=channels, 'B'=batch

% Run ONNX inference ------------------------------------------------
try
    out = predict(models.spk_encoder, inp);
catch ME
    % Try alternative format if first attempt fails
    try
        inp = dlarray(reshape(x_f32, 1, 1, []), 'SCB');
        out = predict(models.spk_encoder, inp);
    catch
        error("extract_speaker_embedding:InferenceFailed", ...
            "x-vector ONNX inference failed: %s\n" + ...
            "Check that models/xvector/xvector_encoder.onnx is valid " + ...
            "and that Deep Learning Toolbox is installed.", ME.message);
    end
end

emb = single(extractdata(out));
emb = emb(:)';                                   % ensure [1 × D_emb]

% Safety check -------------------------------------------------------
if numel(emb) ~= cfg.spk_emb_dim
    error("extract_speaker_embedding:DimMismatch", ...
        "x-vector output has %d dimensions; expected %d. " + ...
        "Verify the ONNX model is the correct checkpoint.", ...
        numel(emb), cfg.spk_emb_dim);
end
end
