# Developer Workflow & Execution Rules  
## Zero-Shot Voice Cloning System (MATLAB Edition)

## The “Local-to-Cloud” Constraint

The developer’s local machine **lacks the resources** to run this project.  
The AI coding agent must strictly adhere to this workflow:

1. **Write Local**  
   - Generate and edit `.m` and `.mlapp` files in the local VS Code workspace.
2. **Git Push**  
   - Push text-based code to GitHub.
3. **Cloud Execution**  
   - The developer will:
     - Pull the code in **MATLAB Online**  
     - Run and test everything there.

## AI Agent Rules for Testing

- **DO NOT** attempt to run MATLAB scripts locally using terminal commands (e.g., `matlab -batch`).  
- **DO NOT** write shell scripts for local testing.  
- Assume all testing happens **asynchronously** by the human developer in MATLAB Online.  

## Data & Git Management

### Storage Constraints

- **MATLAB Drive:** 20 GB limit  
- **GitHub:** File size limits (avoid large binaries)  

### Strict `.gitignore` Rules

Exclude the following from version control:

- All audio files:
  - `*.wav`, `*.mp3`, `*.flac`, `*.ogg`, etc.
- All deep learning weights and model files:
  - `*.onnx`, `*.pt`, `*.bin`, `*.pth`, `*.safetensors`, etc.
- MATLAB binary / auto-save files:
  - `*.mat`, `*.asv`, `*.fig`, `*.mlx`, `*.mlappinstall`, etc.
- Temporary / cache files:
  - `*.log`, `*.tmp`, `__pycache__` (if ever mixed), etc.

### Model Handling

- Do **not** version-control large model files.  
- Implement a utility script:

  - `scripts/download_weights.m`  
    - Uses `websave` (or similar) to download heavy ONNX files **directly** to the MATLAB Drive runtime environment.  
    - Documents expected URLs / sources and target paths within the repo structure.  

All such downloads must be performed by the developer inside MATLAB Online, not by the AI agent locally.

## Git Commit Conventions (Recommended)

Use clear, conventional commit messages, e.g.:

- `feat: add preprocess_signal.m with resampling and noise reduction`  
- `fix: correct STFT windowing in stft_analysis.m`  
- `docs: update PRD with UI playback requirements`  

This keeps the history readable for academic evaluation and future collaborators.