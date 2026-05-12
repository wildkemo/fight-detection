# GEMINI.md - Fight Detection System

## Project Overview
This project implements a real-time **Fight Detection System** designed for efficiency on CPU-only systems and edge devices. The system leverages **YOLOv8-Pose** for simultaneous human detection and skeleton extraction, followed by a **Bidirectional GRU** for temporal classification.

### Main Technologies
- **Language:** Python 3.13
- **Pose Estimation & Tracking:** YOLOv8n-Pose (with ByteTrack)
- **Temporal Modeling:** Bidirectional Gated Recurrent Units (GRU)
- **Deployment:** TensorFlow 2.21
- **Utilities:** OpenCV, NumPy, Scikit-learn

---

## Architecture Pipeline
The system follows a high-performance sequential pipeline:
1.  **Deterministic Frame Sampling:** Input streams are sampled at exactly **12 FPS** using frame skipping to ensure temporal consistency with training data.
2.  **YOLOv8-Pose Extraction:** A single inference pass identifies multiple persons and their 17 body keypoints in global frame coordinates.
3.  **ByteTrack Tracking:** Assigns persistent unique IDs to individuals, preventing "identity teleportation" and ensuring clean motion sequences.
4.  **Interaction Gating:** To save CPU cycles, the GRU is only invoked for persons in close proximity to others or exhibiting high-acceleration motion.
5.  **Bidirectional GRU Inference:** Classifies **36-frame** motion buffers (~3 seconds) using a model that reads the sequence forward and backward for better context.
6.  **Temporal Smoothing (4/6 Rule):** An alert is only triggered if 4 out of the last 6 sequences are flagged as violent, minimizing flickering and false positives.

---

## Building and Running

### Setup
1.  **Environment:** Ensure Python 3.13 is installed.
2.  **Installation:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### Execution
1.  **Preprocessing:** `python -m src.preprocess` (Extracts frames at 12 FPS)
2.  **Pose Extraction:** `python -m src.extract_yolo_poses` (Generates JSON tracks and MP4 visualizations)
3.  **Sequence Building:** `python -m src.build_yolo_sequences` (Creates .npy datasets)
4.  **Training:** `python -m src.train` (Trains the Bidirectional GRU)
5.  **Inference:** `python -m src.inference --source <video_path_or_0>`

---

## Technical Constraints
-   **Sampling Rate:** Strictly 12 FPS (deterministic).
-   **Sequence Length:** 36 frames.
-   **Architecture:** Bidirectional GRU with L2 regularization and 0.5 Dropout.
-   **Detection Threshold:** 0.3 (Recall-optimized).
