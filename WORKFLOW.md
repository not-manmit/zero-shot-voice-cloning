# Developer Workflow & Execution Rules  
## Zero-Shot Voice Cloning System (MATLAB Online Edition)

---

## 1. The "Local-to-Cloud" Architecture

The local development environment lacks the memory and compute resources to execute deep learning models in MATLAB. Therefore, all development follows this strict separation of concerns:

```text
[LOCAL MACHINE / VS CODE]
  ├── Code authoring and editing (.m, .md)
  ├── Static consistency checks & syntax verification
  ├── Git version control & clean commits
  └── PUSH TO GITHUB
            │
            ▼
[MATLAB ONLINE / MATLAB DRIVE]
  ├── PULL REPOSITORY
  ├── run("scripts/setup_matlab_online.m")     <- downloads ~875 MB ONNX models via websave
  ├── run("scripts/validate_matlab_online.m")  <- preflight validation of all toolboxes & models
  ├── run("tests/test_end_to_end.m")           <- synthetic pipeline integrity test
  ├── run("tests/test_end_to_end_real_reference.m") <- real human voice cloning validation
  └── VoiceClonerApp                           <- interactive synthesis UI
```

---

## 2. Strict Execution Rules for AI Coding Agents

- **DO NOT** attempt to run MATLAB locally (`matlab -batch`, `matlab -r`, `matlab -nodisplay`).
- **DO NOT** write bash or PowerShell scripts designed to execute MATLAB locally.
- **DO NOT** claim MATLAB execution has passed unless executed by the human developer in MATLAB Online.
- **NO PYTHON** at runtime: no PyTorch, no FastAPI, no Docker, no ONNX Runtime Python wrappers.

---

## 3. Data & Storage Hygiene

- **MATLAB Drive Quota:** 20 GB standard limit. The 5 canonical ONNX models total ~875.7 MB, easily fitting within storage limits.
- **Git Binary Exclusion:** Model binaries (`*.onnx`, `*.pt`, `*.bin`) and audio waveforms (`*.wav`, `*.mp3`, `*.flac`) are excluded from Git via `.gitignore`.
- Models are downloaded directly inside MATLAB Online using native `websave` in `scripts/download_weights.m`.

---

## 4. Master MATLAB Online Testing Sequence

When testing in MATLAB Online, the developer executes:

```matlab
% 1. Setup paths and download missing ONNX models
run("scripts/setup_matlab_online.m")

% 2. Master preflight validator (checks dependencies, imports, and runs all smoke tests)
run("scripts/validate_matlab_online.m")

% 3. Optional real speech validation (place clean 3-5 s WAV at tests/reference_speech.wav)
run("tests/test_end_to_end_real_reference.m")

% 4. Launch interactive Voice Cloning App
VoiceClonerApp
```