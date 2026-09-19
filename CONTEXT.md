# Project Context: Zero-Shot Voice Cloning System (MATLAB Edition)

## Background

This project is an academic major project for a **Signals & Systems** course.  
The original Python/PyTorch architecture has been **strictly abandoned** due to a professor’s mandate. The entire system must now be implemented in **MATLAB**, leveraging:

- Audio Toolbox  
- Deep Learning Toolbox  
- Signal Processing Toolbox  

## Core Philosophy: “DSP-First” Approach

To satisfy academic evaluation, this is **not** framed as an “AI wrapper.”  
It is an **Acoustic Feature Synthesis Engine driven by Digital Signal Processing (DSP)**.  
AI / neural networks are strictly downstream components.

The implementation must emphasize classic DSP algorithms in the code:

1. **Nyquist–Shannon resampling** to 16 kHz  
2. **Time-domain noise suppression** (e.g., spectral subtraction / gating)  
3. **Short-Time Fourier Transform (STFT)** with Hann windowing  
4. **Linear-to-Mel scale transformation**

## Agent Instructions

- **NO PYTHON**  
  - Do not generate any Python code, FastAPI, Docker, or related artifacts.
- **MATLAB ONLY**  
  - All logic must be written in `.m` scripts, functions, and `.mlapp` (App Designer) files.
- **Terminology**  
  - Use academic Signals & Systems terms in comments and variable names, e.g.:
    - “discrete-time sequence”
    - “spectro-temporal matrix”
    - `fs` for sample rate, `x` for input signal, `y` for output signal, `N` for frame length.