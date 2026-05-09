# GEMINI.md - Fight Detection System

## Project Overview
This project aims to implement a lightweight **Fight Detection System** for real-time video analysis, optimized for CPU-only systems and edge devices like Raspberry Pi. The core approach involves sampling video frames at a low FPS, processing them with a lightweight CNN (**EfficientNet-Lite0**), and using a sliding window aggregation technique to make a final "Fight/No-Fight" decision.

### Main Technologies
- **Language:** Python
- **Model:** EfficientNet-Lite0
- **Frameworks (Proposed):** TensorFlow Lite or ONNX Runtime for optimized inference.
- **Preprocessing:** FFmpeg for frame extraction, OpenCV for image manipulation.

---

## Architecture Pipeline
1. **Video Input:** Raw CCTV or video feeds.
2. **Frame Sampling:** Extract frames at 5 FPS to reduce temporal redundancy.
3. **Preprocessing:** Resize to 224x224, normalize, and apply lightweight denoising/CLAHE.
4. **Model Inference:** Classify individual frames using EfficientNet-Lite0.
5. **Aggregation:** Sliding window (e.g., last 16 frames) to average probabilities.
6. **Decision:** Final binary classification (Fight/Non-Fight) based on a confidence threshold (default: 0.65).

---

## Building and Running

### Setup
- The project uses a Python virtual environment located in `.venv/`.
- **TODO:** Create a `requirements.txt` file listing essential libraries (e.g., `opencv-python`, `tensorflow-lite`, `numpy`).

### Execution
- **TODO:** Implement the frame extraction script.
- **TODO:** Implement the training pipeline using transfer learning.
- **TODO:** Export the trained model to TFLite/ONNX format.

---

## Development Conventions

### Technical Constraints
- **Input Resolution:** Must be 224x224 to match EfficientNet-Lite0 requirements and ensure CPU efficiency.
- **Frame Rate:** Target sampling rate is 5 FPS.
- **Loss Function:** `BCEWithLogitsLoss` for binary classification stability.

### Dataset Structure
- Source videos are stored in `dataset/Violence` and `dataset/NonViolence`.
- **Note:** Always split the dataset by **video** (not frame) before training to prevent data leakage.

### Optimization for Edge (Raspberry Pi)
- Prefer INT8 quantization for deployment.
- Use multithreaded inference where possible.

---

## Key Files
- `SCOPE.md`: Comprehensive implementation plan and architectural details.
- `src/preprocess.py`: (In Progress) Script for frame extraction and preprocessing logic.
- `PREPROCESSING.md`: Documentation for preprocessing steps and rationale.
- `dataset/`: Contains the raw video data (ignored by git).
- `output/`: Directory for generated frames and model checkpoints (ignored by git).
