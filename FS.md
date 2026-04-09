# File Structure (FS.md)

This document describes the organization of the **CCTV Violence Detection System** project.

```text
.
├── .gitignore               # Standard exclusions for Python and Deep Learning
├── GEMINI.md                # High-level project overview and setup
├── PREPROCESSING.md         # Detailed preprocessing logic and steps
├── SCOPE.md                 # Project goals, objectives, and approach
├── FS.md                    # File structure documentation (this file)
└── src/                     # All source code
    ├── data/                # Data processing and preprocessing scripts
    │   ├── __init__.py
    │   ├── extract_frames.py # Extracts frames from video files at fixed FPS
    │   ├── normalize.py      # Resizes and normalizes frame pixel values
    │   └── sequence_builder.py # Groups frames into temporal sequences
    ├── evaluation/          # Scripts for calculating evaluation metrics
    │   ├── __init__.py
    │   └── metrics.py        # Precision, Recall, and F1-score implementation
    ├── models/              # Deep learning model architectures
    │   ├── __init__.py
    │   └── cnn_lstm.py       # Initial CNN + LSTM implementation
    ├── evaluate.py          # Main entry point for evaluating a trained model
    ├── inference.py         # Script for real-time inference (includes webcam support)
    └── train.py             # Main entry point for training the model
```

## Key Directories

- **src/data/**: Handles the entire pipeline from raw video to ready-to-train tensors.
- **src/models/**: Contains the neural network architectures used for detection.
- **src/evaluation/**: Centralized location for custom metrics critical for violence detection (minimizing false positives).
- **src/** (root): Entry points for high-level operations like `train`, `evaluate`, and `inference`.
