# CPU-Optimized Fight Detection Pipeline (Interaction-Aware)

This project implements a multi-stage pipeline for robust fight detection in CCTV-style videos, prioritizing **pairwise interactions** and **motion-based features** over static single-person poses.

## 🏗️ Architecture Overview

The system follows a 5-stage decoupled pipeline:
1.  **`detect_and_track.py`**: YOLOv8n (person only) + BoT-SORT/ByteTrack. Focus on stable IDs.
2.  **`extract_pose.py`**: RTMPose on cropped bboxes. Must convert crop coords to full-frame coords.
3.  **`build_sequences.py`**: Feature engineering stage. Converts raw skeletons into normalized interaction features.
4.  **`train_tcn.py`**: Residual Dilated Temporal Convolutional Network (TCN).
5.  **`inference.py`**: Real-time integration with proximity-based pose optimization.

## 🛠️ Technical Mandates & Conventions

### 1. Interaction-First Modeling
*   **Mandate**: Never train on isolated single-person sequences. All training samples MUST represent interactions (Pairwise: Person A + Person B).
*   **Context**: Fights are relational events. Single-person "aggressive" motion often causes false positives in crowded or high-activity scenes.

### 2. Skeleton Normalization
*   **Formula**: `kpts_norm = (kpts - pelvis_center) / torso_length`.
*   **Requirement**: All features must be scale-invariant and translation-invariant to handle varying camera distances and resolutions.

### 3. Feature Engineering (The "Strong Signal" Rule)
Every sequence sample (36 frames) must include:
*   **Motion**: Velocity ($v$) and Acceleration ($a$) per joint.
*   **Energy**: Intensity of movement ($\sum ||kpts_t - kpts_{t-1}||^2$).
*   **Interaction**: Inter-person distance, closing speed, and relative velocity.
*   **Geometry**: Critical joint distances (e.g., hand-to-head, wrist-to-torso).

### 4. Data Integrity
*   **Contiguity**: Only build sequences from contiguous frames. If a track has a gap, split the sequence.
*   **Confidence**: Mask or zero-fill keypoints with confidence < 0.3.

### 5. Training Standards
*   **Weights**: Use balanced class weights. Do NOT artificially inflate the positive class weight (e.g., avoid `class_weights[1] *= 3.0`) as it leads to excessive false positives.
*   **Thresholding**: Never hardcode a low threshold (like 0.3). Always tune the decision threshold using validation data to optimize the F1-score/Precision-Recall balance.
*   **Architecture**: Use **Dilated Convolutions** and **Residual Blocks**. Avoid aggressive MaxPooling that might erase short-duration motion bursts (punches/impacts).

## 🚀 Optimization Strategy
*   **Proximity Filter**: In `inference.py`, only run RTMPose if `distance(Person_A, Person_B) < Interaction_Threshold`. This ensures CPU feasibility by skipping pose estimation for isolated individuals.

## 📁 Directory Structure
*   `data/tracks/`: JSON files with bbox and IDs.
*   `data/poses/`: JSON files with RTMPose skeletons (full-frame coords).
*   `data/sequences/`: Preprocessed `.npy` files for training.
*   `models/`: ONNX (Detection/Pose) and TFLite (TCN) models.
