# GEMINI.md - Fight Detection System

## Project Overview
This project implements a real-time **Fight Detection System** designed for efficiency on CPU-only systems and edge devices. The system leverages **Pose Estimation** and **Temporal Modeling (GRU)** to identify aggressive movements (punches, kicks, etc.) while ignoring background noise. By focusing on human keypoints rather than raw pixel data, the model remains lightweight and robust to varying environments.

### Main Technologies
- **Language:** Python
- **Pose Estimation:** MoveNet Lightning (via TensorFlow Hub)
- **Object Detection:** YOLOv8n (for person localization)
- **Sequence Modeling:** Gated Recurrent Units (GRU)
- **Deployment:** TensorFlow Lite (INT8 Quantization)
- **Utilities:** OpenCV, NumPy, TensorFlow Lite

---

## Architecture Pipeline
The system follows a sequential pipeline to process video data:
1.  **Frame Sampling:** Videos are sampled at **5 FPS** to reduce redundancy and CPU load while preserving motion information.
2.  **Person Detection:** YOLOv8n identifies and crops persons within the frame to focus the pose estimator.
3.  **Pose Estimation:** MoveNet Lightning extracts 17 body keypoints (x, y, confidence) for each detected person.
4.  **Temporal Sequencing:** Keypoints from **16 consecutive frames** (~3 seconds of context) are aggregated into a motion buffer.
5.  **GRU Inference:** A GRU-based neural network classifies the sequence as "Fight" or "No-Fight".
6.  **Decision Smoothing:** A temporal voting logic (e.g., 8/12 positive predictions) triggers an alert to minimize false positives.

---

## Building and Running

### Setup
1.  **Environment:** The project uses a Python virtual environment in `.venv/`.
2.  **Dependencies:** (TODO) Create `requirements.txt`. Essential libraries include:
    - `ultralytics` (YOLOv8)
    - `tensorflow` / `tensorflow-hub`
    - `opencv-python`
    - `numpy`

### Execution
-   **TODO:** Implement `src/preprocess.py` for frame extraction and dataset splitting.
-   **TODO:** Implement `src/train.py` for the GRU model training.
-   **TODO:** Implement `src/inference.py` for real-time webcam detection.
-   **TODO:** Create an export script for TFLite conversion.

---

## Development Conventions

### Technical Constraints
-   **Sampling Rate:** Strictly 5 FPS for both training and inference.
-   **Sequence Length:** 16 frames (816-feature vector after flattening 17x3x16).
-   **Input Resolution:** YOLOv8/MoveNet handle resizing, but target resolution for detection is typically 640px (YOLO) and 192px (MoveNet).

### Dataset Structure
-   Source videos are located in `dataset/Violence` and `dataset/NonViolence`.
-   **Crucial Rule:** Always split the dataset by **video ID**, never by frames, to prevent data leakage and inflated accuracy.

### Optimization for Edge
-   Use **INT8 Quantization** during TFLite conversion.
-   Prefer MoveNet "Lightning" over "Thunder" for better performance on weak CPUs.

---

## Key Files
-   `SCOPE.md`: Comprehensive implementation plan and architectural details (Primary Reference).
-   `PREPROCESSING.md`: Documentation for data preparation steps.
-   `src/`: Directory for source code (Pipeline implementation in progress).
-   `dataset/`: Raw video data (ignored by git).
-   `output/`: Generated frames and model checkpoints (ignored by git).
