# Real-Time Fight Detection System — Implementation Plan

## 1. Dataset Split

### Goal

Split the dataset into:

- train
- validation
- test

### Important

Split by VIDEO, not by frames.

Correct:

```text
video_001.mp4 → train
video_002.mp4 → test
```

Wrong:

```text
same video's frames in both train and test
```

This prevents data leakage and fake accuracy.

---

## 2. Frame Extraction

### Goal

Extract frames from videos at 5 FPS.

### FFmpeg Command

```bash
ffmpeg -i input.mp4 -vf fps=5 frames/frame_%04d.jpg
```

### How it works

- `-vf fps=5`
  → extracts 5 frames every second
- `frame_%04d.jpg`
  → saves sequential frame images

### Why 5 FPS?

- reduces redundancy
- lowers CPU cost
- still preserves fight motion information

---

## 3. Person Detection

### Goal

Detect people inside each frame.

### Recommended Model

- YOLOv8n

### Output

Bounding boxes around detected persons.

Example:

```text
Person 1 → x1,y1,x2,y2
Person 2 → x1,y1,x2,y2
```

### Why this step matters

- removes background noise
- focuses only on human interaction
- improves accuracy significantly

---

## 4. Pose Estimation

### Goal

Extract human body keypoints from detected persons.

### Recommended Model

- MoveNet Lightning

### Output

17 body keypoints:

```text
nose
shoulders
elbows
wrists
hips
knees
ankles
...
```

Each keypoint contains:

```text
x coordinate
y coordinate
confidence score
```

### Why pose estimation helps

The system learns:

- punches
- kicks
- aggressive movement patterns

instead of learning background appearance.

---

## 5. Build Temporal Sequences

### Goal

Create motion sequences from consecutive frames.

### Recommended Window

- 16 frames

At:

```text
5 FPS
```

this gives:

```text
~3 seconds of motion context
```

### Example

Instead of:

```text
1 image
```

you train on:

```text
16 consecutive pose frames
```

---

## 6. Flatten Pose Features

### Original Shape

```text
(16, 17, 3)
```

Meaning:

- 16 frames
- 17 keypoints
- x/y/confidence

### Flattened Shape

```text
(16, 51)
```

because:

```text
17 × 3 = 51
```

### Why flatten?

GRU layers expect sequential feature vectors.

---

## 7. Train GRU Model

### Goal

Learn motion behavior over time.

### Recommended Architecture

```python
model = Sequential([
    GRU(64, input_shape=(16, 51)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

### How it works

#### GRU Layer

Learns:

- movement patterns
- motion timing
- aggressive sequences

#### Dense Layer

Learns higher-level fight features.

#### Sigmoid Output

Returns:

```text
0 → nonfight
1 → fight
```

---

## 8. Train on Sequences

### Training Data Format

```text
X = pose sequences
Y = labels
```

### Example Shapes

```text
X shape:
(12000, 16, 51)

Y shape:
(12000,)
```

### Explanation

- each sample = one motion sequence
- label = fight or nonfight

---

## 9. Real-Time Inference (Laptop Camera)

### Goal

Run live detection using laptop webcam.

### Pipeline

```text
Laptop Camera
      ↓
Frame Sampling
      ↓
YOLOv8n
      ↓
MoveNet
      ↓
Sequence Buffer
      ↓
GRU Model
      ↓
Fight Detection
```

### OpenCV Webcam Example

```python
import cv2

cap = cv2.VideoCapture(0)
```

### How it works

- `0`
  → default laptop webcam

---

### Read Frames

```python
ret, frame = cap.read()
```

### Frame Sampling

Process only:

```text
5 FPS
```

instead of every frame.

---

### Sequence Buffer Logic

Store:

```text
last 16 pose vectors
```

When buffer is full:

- run GRU inference
- output confidence score

---

## 10. Smart Temporal Decision Logic

### Goal

Reduce false positives.

### Recommended Strategy

Do NOT trigger on single prediction.

Use:

- confidence smoothing
- consecutive detections

### Example Rule

```text
8 of last 12 predictions > 0.8
```

before raising an alert.

### Why this works

Prevents:

- random spikes
- accidental aggressive poses
- noisy predictions

---

## 12. Export to TensorFlow Lite

### Goal

Optimize deployment for weak devices.

### Convert Model

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
```

### Save File

```python
with open("fight_detector.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## INT8 Quantization (Recommended)

### Enable Optimization

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

### Benefits

- smaller model
- faster CPU inference
- lower RAM usage

### Final Output

```text
fight_detector.tflite
```

This is the optimized deployment model for:

- weak PCs
- Raspberry Pi
- edge AI systems
