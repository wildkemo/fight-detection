# File Structure (FS.md)

This document describes the organization of the **CCTV Violence Detection System** project.

```text
.
├── .gitignore               # Standard exclusions
├── GEMINI.md                # High-level project overview
├── PREPROCESSING.md         # Detailed preprocessing logic and steps
├── SCOPE.md                 # Project goals and objectives
├── FS.md                    # File structure documentation (this file)
└── src/
    └── data/                # Vision-focused modular preprocessing
        ├── color_space.py    # Step 7: Color space optimization
        ├── contrast.py       # Step 5: CLAHE contrast enhancement
        ├── denoise.py        # Step 3: Spatial noise reduction
        ├── dynamic_range.py  # Step 4: Log/Inverse-Log transformations
        ├── edge_enhancement.py # Step 6: Sharpening and boundary detection
        ├── extract_frames.py # Step 1: Video frame extraction
        ├── preprocess.py     # Main orchestrator for the 9-step pipeline
        ├── resize.py         # Step 2: Uniform resizing
        ├── sanity_check.py   # Step 8: Visual inspection and validation
        └── storage.py        # Step 9: Organized storage and burst grouping
```

## Key Directories

- **src/data/**: Contains the modular implementation of the 8-step image enhancement pipeline. Each file corresponds to a specific step defined in `PREPROCESSING.md`.
- **preprocess.py**: The central script that imports and executes all 8 preprocessing steps in sequence.
