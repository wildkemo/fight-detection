# CNN Implementation Flaws & Improvements

This document outlines identified flaws and potential improvements for the Violence Detection CNN+LSTM model.

## 1. Architectural Flaws (`src/models/cnn_lstm.py`)

### GPU Performance Bottleneck
*   **Issue:** The use of `recurrent_dropout=0.3` in the `LSTM` layer.
*   **Impact:** In TensorFlow/Keras, `recurrent_dropout` prevents the model from using the highly optimized **cuDNN** kernels. This can make training **5x to 10x slower** on a GPU.
*   **Fix:** Remove `recurrent_dropout` and use standard `Dropout` layers between the LSTM and Dense layers.

### Information Bottleneck
*   **Issue:** `GlobalAveragePooling2D()` collapses the entire $14 \times 14$ spatial grid into a single vector.
*   **Impact:** This may discard spatial relationships crucial for detecting violence (e.g., the specific position of a person relative to another).
*   **Improvement:** Experiment with `Flatten()` or `MaxPooling2D` to retain more spatial context if accuracy is low.

### Shallow Feature Extractor
*   **Issue:** A 3-layer CNN is relatively shallow for complex video classification.
*   **Improvement:** Consider using a pre-trained backbone (like MobileNetV2) or adding 1-2 more convolutional blocks to capture more robust features.

## 2. Data Pipeline Flaws (`src/data/sequence_builder.py`)

### RAM Usage Risk (Memory Error)
*   **Issue:** `build_dataset()` loads the **entire dataset** into a single NumPy array (`X.append(sequence)`).
*   **Impact:** Video data is 5D (Videos, Frames, Height, Width, Channels). This will cause a **Memory Error (OOM)** as the dataset grows.
*   **Fix:** Implement a `tf.data.Dataset` generator or a Keras `Sequence` to load videos in batches from disk.

### Color Space Inconsistency
*   **Issue:** Three-way contradiction: `config.py` specifies RGB, `PREPROCESSING.md` aims for Grayscale, but `sequence_builder.py` uses `cv2.imread` (BGR) without conversion.
*   **Impact:** The model trains on BGR data while configuration and documentation expect different formats, leading to potential issues with pre-trained backbones.
*   **Fix:** Standardize on one format (e.g., RGB or Grayscale) across all modules.

### Redundant Processing
*   **Issue:** `sequence_builder.py` performs `cv2.resize` on frames that were already resized during the `preprocess.py` stage.
*   **Impact:** Unnecessary CPU overhead during training initialization.
*   **Fix:** Trust the preprocessed frame dimensions or move all resizing to the data generator.

### Naive Padding Strategy
*   **Issue:** Shorter videos are padded by repeating the last frame.
*   **Impact:** This creates "frozen" motion at the end of clips, which can confuse the LSTM's temporal logic.
*   **Improvement:** Use zero-padding or mirror padding, and mask the padded steps in the LSTM layer.

### Unused Configuration Parameters
*   **Issue:** `FRAME_STRIDE = 2` is defined in `config.py` but ignored by `sequence_builder.py`, which uses its own `np.linspace` sampling logic.
*   **Impact:** Misleading configuration file.

### Lack of Augmentation
*   **Issue:** No spatial (flipping, rotation) or temporal (random starting frames) augmentation is applied.
*   **Impact:** High risk of overfitting, especially on small surveillance datasets.
*   **Fix:** Add random horizontal flipping and brightness adjustments during frame loading.

## 3. Training & Logic Flaws (`src/training/train.py`)

### Configuration Fragility
*   **Issue:** `src/data/preprocess.py` uses a hardcoded `FRAME_SIZE = (112, 112)` instead of importing it from `config.py`.
*   **Impact:** Changes to `config.py` will not reflect in the actual preprocessing pipeline, leading to shape mismatches.

### Batch Size Stability
*   **Issue:** `BATCH_SIZE = 4` is very small.
*   **Impact:** Can lead to noisy gradients and unstable training.
*   **Improvement:** If memory allows, increase to 8 or 16, or use Gradient Accumulation.

### Inefficient Dataset Splitting
*   **Issue:** The dataset is split *after* loading everything into RAM.
*   **Impact:** Contributes to the memory bottleneck.
*   **Fix:** Split the file paths (metadata) first, then use a generator to load only the required split.

### Lack of Class Weighting
*   **Issue:** No handling for class imbalance (Violence vs. Non-Violence).
*   **Impact:** The model may become biased toward the majority class.
*   **Fix:** Calculate and pass `class_weight` to `model.fit()`.

### Lack of Advanced Normalization
*   **Issue:** Only dividing by `255.0` is used.
*   **Improvement:** Use standard normalization (mean subtraction and standard deviation scaling) to speed up convergence.
