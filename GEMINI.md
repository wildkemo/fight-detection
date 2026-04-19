# Project Overview

This is a **CCTV Violence Detection Preprocessing System** designed to enhance surveillance video segments to make fighting activities and edges more visible.

The system focuses on spatial (appearance) and structural (motion boundary) enhancement. The main challenges addressed include dealing with low-quality CCTV footage and varying lighting conditions.

# Project Structure

```text
src/
└── data/
    ├── preprocess.py          # Orchestrates the 8-step preprocessing pipeline
    ├── extract_frames.py      # Step 1: Video frame extraction
    ├── resize.py              # Step 2: Uniform resizing
    ├── denoise.py             # Step 3: Spatial noise reduction
    ├── dynamic_range.py       # Step 4: Log/Inverse-Log transformations
    ├── contrast.py            # Step 5: CLAHE contrast enhancement
    ├── edge_enhancement.py    # Step 6: Sharpening and boundary detection
    ├── color_space.py         # Step 7: Color space optimization
    └── sanity_check.py        # Step 8: Visual validation
```

# Development Conventions & Architecture

## Preprocessing Pipeline
As outlined in `PREPROCESSING.md`, the data processing pipeline involves:
1.  **Frame Extraction:** Sampling video frames at 5–10 FPS.
2.  **Resizing:** Standardizing frame dimensions (e.g., 224x224).
3.  **Dynamic Range Adjustment:** Applying Log or Inverse-Log transformations based on brightness.
4.  **Contrast Enhancement:** Using CLAHE to improve local contrast and subject visibility.
5.  **Edge Enhancement:** Applying sharpening filters to make physical interactions and movement boundaries distinct.

## Focus
Preprocessing must prioritize making human subjects clearly visible and their movement-defining edges sharp.
