# Fight Detection System — Implementation Plan

## 1. Overall Architecture

Pipeline:

Video → Frame Sampling (5 FPS) → Preprocessing → EfficientNet-Lite → Sliding Window Aggregation → Fight/No-Fight Decision

### Description

- The system processes sampled video frames individually using a lightweight CNN.
- Temporal understanding is achieved using prediction aggregation over multiple frames instead of heavy video models.
- This keeps inference fast enough for CPU-only systems.

---

## 2. Dataset Preparation

### Description

- Extract frames from videos at 5 FPS.
- Split dataset by VIDEO before frame extraction to prevent data leakage.
- Organize frames into:
  - train/
  - validation/
  - test/

### Example FFmpeg Command

```bash
ffmpeg -i input.mp4 -vf fps=5 output/frame_%04d.jpg
```

---

## 4. Preprocessing

### Description

Apply lightweight preprocessing only.

### Recommended

- Resize
- Normalization
- Mild denoising
- Optional CLAHE

### Avoid

- Heavy sharpening
- Aggressive edge filters
- Complex transformations

### Reason

EfficientNet already learns strong visual features.

---

## 5. Model Choice

### Model

- EfficientNet-Lite0

### Description

- Lightweight and CPU-friendly
- Better feature extraction than MobileNet in many cases
- Good balance between speed and accuracy

---

## 6. Input Resolution

### Resolution

- 224×224

### Description

- Reduces CPU usage significantly
- Maintains good visual detail for classification
- Standard size for EfficientNet-Lite

---

## 7. Training Strategy

### Description

Use transfer learning instead of training from scratch.

### Steps

1. Load pretrained ImageNet weights
2. Replace final classifier layer
3. Train classifier head first
4. Fine-tune full model with low learning rate

### Reason

Pretrained models generalize much better on medium-sized datasets.

---

## 8. Data Augmentation

### Recommended Augmentations

- Brightness variation
- Slight blur
- Noise
- Compression artifacts
- Small rotations

### Avoid

- Extreme crops
- Heavy color modifications
- Unrealistic augmentations

### Reason

Augmentations should simulate real CCTV conditions.

---

## 9. Loss Function

### Loss

- BCEWithLogitsLoss

### Description

- Suitable for binary classification
- More numerically stable than sigmoid + BCE separately

---

## 10. Inference Logic

### Description

- Each frame receives a fight probability score from the model.
- Predictions are accumulated over time before making a final decision.

### Example Output

- 0.92 → fight
- 0.15 → non-fight
- 0.81 → fight

---

## 11. Sliding Window Aggregation

### Description

- Store predictions from recent frames.
- Compute average probability over a fixed window.

### Example

- Last 16 frames
- Average confidence used for final decision

### Benefits

- Reduces noisy predictions
- Stabilizes output
- Mimics temporal understanding without expensive models

---

## 12. Suggested Parameters

### Recommended

- FPS: 5
- Window size: 16 frames
- Threshold: 0.65

### Description

- 16 frames at 5 FPS ≈ 3 seconds of temporal context
- Threshold can be tuned later using validation data

---

## 14. Deployment Format

### Recommended

- TensorFlow Lite
  OR
- ONNX Runtime

### Description

- Optimized for lightweight inference
- Suitable for Raspberry Pi and weak CPUs
- Easier deployment than full PyTorch models

---

## 15. Raspberry Pi Optimization

### Recommended Optimizations

- INT8 quantization
- Multithreaded inference
- Frame skipping when overloaded

### Description

These optimizations reduce:

- CPU usage
- Memory usage
- Inference latency

while maintaining acceptable accuracy.
