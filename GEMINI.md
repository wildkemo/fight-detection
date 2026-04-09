# Project Overview

This is a deep learning **CCTV Violence Detection System** designed to classify surveillance video segments into two categories: **violence** and **non-violence**. 

The system focuses on learning spatial (appearance) and temporal (motion) patterns to detect fights and aggressive interactions. The planned modeling approach starts with a combination of **CNN and LSTM**, with potential future improvements using 3D CNNs or Transformers. The main challenges addressed include dealing with low-quality CCTV footage, noisy labels, and dataset bias.

# Project Structure

```text
src/
├── data/
│   ├── extract_frames.py      # Extracts frames from videos at specified FPS
│   ├── normalize.py           # Resizes and normalizes pixel values
│   └── sequence_builder.py    # Groups frames into sequences (16-30 frames)
├── models/
│   └── cnn_lstm.py            # Initial CNN + LSTM model implementation
├── evaluation/
│   └── metrics.py             # Precision, Recall, and F1-score calculations
├── train.py                   # Main training loop
├── evaluate.py                # Script for evaluating the trained model
└── inference.py               # Script for running inference (potentially real-time)
```

# Building and Running

The project is currently in its early stages and relies on a Python virtual environment. 

To run the basic webcam capture script:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the vision script
python src/inference.py
```

*Note: The project requires `opencv-python` which seems to be used in the `vision.py` script.*

# Development Conventions & Architecture

## Preprocessing Pipeline
As outlined in `PREPROCESSING.md`, the data processing pipeline is strict and involves:
1.  **Frame Extraction:** Sampling video frames at 5–10 FPS to capture motion while reducing redundancy.
2.  **Resizing:** Normalizing frame dimensions to 224×224.
3.  **Pixel Normalization:** Scaling pixel values from [0–255] to [0–1].
4.  **Sequence Construction:** Grouping frames into sequences of 16–30 frames to capture temporal motion context.
5.  **Data Organization:** Processed data must be organized into strict directories (`dataset/train/violence`, `dataset/test/non_violence`, etc.) to prevent data leakage.

## Modeling and Evaluation
-   **Metrics:** The primary metrics for evaluating the system are Precision, Recall, and the F1-score (which is critical for minimizing false positives in violence detection).
-   **Focus:** Preprocessing and modeling should prioritize the underlying learning signal (motion and human interaction) over superficial visual quality.