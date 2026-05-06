# Implementation Plan: Fixing CNN & Data Pipeline Flaws

This document outlines a detailed, step-by-step plan to resolve the issues identified in `FLAWS.md`.

## Phase 1: Eliminate the Memory Bottleneck (Data Pipeline)
**Target Flaws:** RAM Usage Risk (OOM), Inefficient Dataset Splitting, Unused Configuration Parameters

*   **Step 1.1: Metadata Extraction:** Rewrite the data loader in `sequence_builder.py` to only scan directories and build a lightweight list of video folder paths and their labels (0 or 1).
*   **Step 1.2: Early Splitting:** Modify `train.py` to apply `train_test_split` directly to these lists of file paths. This ensures we don't load validation/test images during the training phase.
*   **Step 1.3: Lazy Loading with `tf.data`:** Implement a `tf.data.Dataset` pipeline. This will load physical images from the drive *on-the-fly*, batch by batch, only when requested by the GPU.
*   **Step 1.4: Respect Sampling Config:** Ensure the `tf.data` generator respects the `FRAME_STRIDE` parameter when selecting frames for a sequence.

## Phase 2: Improve Preprocessing and Generalization (Data Pipeline)
**Target Flaws:** Naive Padding Strategy, Lack of Augmentation, Lack of Advanced Normalization, Color Space Inconsistency, Redundant Processing

*   **Step 2.1: Smart Padding:** Update the sequence builder to use zero-padding (black frames) instead of repeating the final frame.
*   **Step 2.2: Sequence-level Augmentation:** Integrate a mapping function into the `tf.data` pipeline to apply random horizontal flipping. Flips must be applied consistently to *every frame* in a video sequence to preserve temporal logic.
*   **Step 2.3: Advanced Normalization:** Update the normalization logic to use dataset-wide mean subtraction and standard deviation scaling (StandardScaler approach) instead of just `1/255.0`.
*   **Step 2.4: Color Space Unification:** Enforce a single color space (standardizing on RGB or Grayscale as per docs) consistently across `config.py`, `sequence_builder.py`, and the model input.
*   **Step 2.5: Eliminate Redundant Resizing:** Remove the redundant `cv2.resize` operation in the data loader, relying on the sizes already established in the preprocessing stage.

## Phase 3: Optimize Model Architecture and Speed (CNN+LSTM)
**Target Flaws:** GPU Performance Bottleneck, Information Bottleneck, Shallow Feature Extractor

*   **Step 3.1: cuDNN Acceleration:** Remove `recurrent_dropout=0.3` from the LSTM layer in `cnn_lstm.py` to enable optimized NVIDIA cuDNN kernels. Add a standard `Dropout` layer immediately after the LSTM for regularization.
*   **Step 3.2: Retain Spatial Context:** Replace `GlobalAveragePooling2D` with a `Flatten` layer (or standard `MaxPooling2D` layers) to preserve the spatial layout of features for the LSTM.
*   **Step 3.3: Deepen the Backbone:** Add a fourth Convolutional block (e.g., 256 filters) to the CNN feature extractor to capture more complex patterns.

## Phase 4: Stabilize Training Configuration (Training Logic)
**Target Flaws:** Batch Size Stability, Configuration Fragility, Lack of Class Weighting

*   **Step 4.1: Increase Batch Size:** Update `config.py` to increase `BATCH_SIZE` from `4` to `8` or `16`, now that the RAM bottleneck is removed.
*   **Step 4.2: Lightweight Test Exports:** Modify `train.py` to save only test file paths (`test_metadata.npz`) instead of raw image arrays, saving significant disk space.
*   **Step 4.3: Centralize Configuration:** Refactor `src/data/preprocess.py` to import `FRAME_SIZE` and other parameters directly from `src/utils/config.py`, removing hardcoded overrides.
*   **Step 4.4: Automated Class Weighting:** Add logic to `train.py` to calculate class weights based on the training split distribution and pass them to `model.fit()`.
