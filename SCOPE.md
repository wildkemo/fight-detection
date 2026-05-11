# Fight Detection System - Final Scope & Implementation Summary

## 1. Project Objective
Implement a robust, real-time violence detection system optimized for edge devices and CPU-only environments using a high-recall tiered pipeline.

## 2. Final Architecture (v2)
The project pivoted from a fragmented YOLO+MoveNet approach to a unified **YOLOv8-Pose** pipeline to resolve identity teleportation and "Frankenstein skeleton" issues.

### Stage 1: Preprocessing & Sampling
- **Sampling:** 5 FPS (deterministic frame skipping).
- **Split:** Stratified by Video ID (80% Train, 10% Val, 10% Test).

### Stage 2: Unified Tracking & Pose Estimation
- **Model:** YOLOv8n-Pose (Nano).
- **Tracking:** ByteTrack (Multi-object tracking).
- **Feature Set:** 17 body keypoints [x, y, confidence] in global frame coordinates.

### Stage 3: Temporal Classification
- **Architecture:** Bidirectional GRU (64 units) + Dense (32 units).
- **Sequence Length:** 16 frames (~3 seconds).
- **Regularization:** L2 Kernel Regularization (0.01) + Dropout (0.5) + Recurrent Dropout (0.2).

### Stage 4: Real-Time Inference (Optimized)
- **FPS Control:** Deterministic Nth-frame sampling.
- **Interaction Gating:** GRU inference is only triggered if two people are within 300px (Interaction Distance).
- **Temporal Smoothing:** 3-out-of-5 sliding window voting logic to trigger "FIGHT!" alerts.

## 3. Performance Summary
- **Recall (Violence):** 94.1%
- **Precision (Violence):** 57.9%
- **Overall Accuracy:** 74.8%
- **Zero-Miss Goal:** Achieving >94% recall ensures that nearly all violent incidents are captured, satisfying the high-security requirement.

## 4. Usage Guide
The system is executed through the following sequence:

| Step | Script | Description |
| :--- | :--- | :--- |
| 1 | `src/preprocess.py` | Extracts frames and prepares the directory structure. |
| 2 | `src/extract_yolo_poses.py` | Performs YOLO-Pose tracking and saves JSON data. |
| 3 | `src/build_yolo_sequences.py` | Converts JSON tracks into .npy training sequences. |
| 4 | `src/train.py` | Trains the Bidirectional GRU model. |
| 5 | `src/inference.py` | Runs the real-time detection system. |

---
*Status: All implementation steps completed and verified.*
