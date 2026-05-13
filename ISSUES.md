# Fight Detection Pipeline Review & Fixes

---

# Root Problem

The current pipeline is:

```text
single tracked person
    ↓
36-frame pose sequence
    ↓
flatten to (36, 51)
    ↓
TCN
    ↓
fight/non-fight
```

This has a major limitation:

> The model never actually sees interactions between people.

So the network starts learning shortcuts like:

- people standing close together
- crowded scenes
- noisy pose estimation
- overlapping skeletons
- camera motion

instead of learning actual fight behavior.

---

# 1) extract_poses.py

---

## Issue → Tracking instability during occlusion / overlap

### Problem

The tracker can switch IDs when:

- people overlap
- occlusion happens
- bodies cross each other
- fast motion occurs

This happens frequently during fights.

So one sequence may accidentally contain:

```text
frames 1–18  → person A
frames 19–36 → person B
```

which destroys temporal consistency.

---

### Fix

Process frames as a continuous stream and preserve tracker state as much as possible.

The goal is:

```text
one track_id = one real human over time
```

---

### Why this matters

Temporal models assume:

```text
same entity across all frames
```

ID switches create fake motion patterns.

---

## Issue → Low-confidence keypoints treated as valid

### Problem

All keypoints are currently saved directly, even weak or incorrect ones.

When people stand close together:

- arms overlap
- shoulders swap
- joints jitter
- detections become noisy

The classifier then learns noise instead of motion.

---

### Fix

Filter or mark low-confidence joints.

Example logic:

```python
if confidence < 0.3:
    treat joint as missing
```

Possible approaches:

- zero-fill bad joints
- keep confidence values
- add missing masks

---

### Why this matters

The model must distinguish:

```text
real motion
vs
pose estimation noise
```

---

## Issue → Missing contextual tracking information

### Problem

Only these are stored:

- frame_id
- keypoints

This is insufficient for interaction reasoning.

---

### Fix

Also store:

- bounding box
- center coordinates
- person size
- average pose confidence

Example:

```python
{
    "frame_id": frame_idx,
    "bbox": [...],
    "center_x": ...,
    "center_y": ...,
    "pose_conf": ...
}
```

---

### Why this matters

Later you can compute:

- movement speed
- inter-person distance
- relative velocity
- collision behavior

which are critical for fight detection.

---

## Issue → JSON structure is too track-centric

### Problem

Data is organized only by track_id.

Fight detection needs both:

- temporal tracking
- frame-level interactions

---

### Fix

Store both:

```python
{
    "frames": {...},
    "tracks": {...}
}
```

---

### Why this matters

Frame-based access allows:

- nearest-person lookup
- pairwise interaction modeling
- crowd context analysis

---

## Issue → No corrupted-frame handling

### Problem

If `cv2.imread()` fails, processing crashes.

---

### Fix

Check:

```python
if img is None:
    continue
```

---

### Why this matters

Single corrupted frames should not stop preprocessing.

---

# 2) build_sequences.py

---

## Issue → Training on single persons instead of interactions

### Problem

This loop:

```python
for track_id, instances in video_tracks.items():
```

creates sequences for only ONE person.

But fights are:

```text
interaction events
```

not isolated single-person actions.

---

### Fix

Build pairwise interaction sequences.

Each sample should include:

- person A pose
- person B pose
- relative distance
- relative velocity
- interaction features

---

### Why this matters

The model needs to learn:

- approaching behavior
- collision-like movement
- reciprocal aggression
- fast mutual motion

instead of:

```text
"person exists near another person"
```

---

## Issue → Raw coordinates are not normalized

### Problem

Current code:

```python
flattened_window = np.array(window).reshape(SEQUENCE_LENGTH, 51)
```

uses raw image coordinates.

This causes the model to learn:

- camera position
- zoom level
- person distance from camera
- image layout

instead of body motion.

---

### Fix

Normalize every skeleton.

---

### Suggested normalization

#### Step 1 — Compute body center

Use hip center:

```python
center_x = (left_hip_x + right_hip_x) / 2
center_y = (left_hip_y + right_hip_y) / 2
```

---

#### Step 2 — Compute body scale

Example:

```python
scale = shoulder_distance
```

---

#### Step 3 — Normalize joints

```python
x_norm = (x - center_x) / scale
y_norm = (y - center_y) / scale
```

---

### Why this matters

Normalization removes dependence on:

- camera distance
- image resolution
- body size in frame

allowing the model to focus on:

```text
pose geometry + motion
```

---

## Issue → No motion features

### Problem

Only positions are used.

Static positions are weak for fight detection.

---

### Fix

Add temporal velocity features.

For each joint:

```python
dx = x_t - x_(t-1)
dy = y_t - y_(t-1)
```

---

### Recommended feature set

Per keypoint:

```text
x_norm
y_norm
confidence
dx
dy
```

---

### Why this matters

Fights contain:

- sudden acceleration
- chaotic motion
- repeated fast limb movement

Velocity is one of the strongest signals.

---

## Issue → Sliding windows overlap too heavily

### Problem

Current stride:

```python
stride = 1
```

creates almost identical samples.

Example:

```text
1–36
2–37
3–38
```

---

### Fix

Use larger stride.

Example:

```python
STRIDE = 6
```

---

### Why this matters

Reduces:

- overfitting
- memorization
- inflated validation accuracy

while keeping temporal coverage.

---

## Issue → Non-contiguous frame sequences

### Problem

Tracks may contain frame gaps:

```text
1,2,3,10,11,12
```

Current logic still builds sequences across gaps.

---

### Fix

Only build windows from contiguous frames.

Check:

```python
frame_id[i+1] == frame_id[i] + 1
```

---

### Why this matters

Temporal models assume continuous motion.

Frame jumps create fake dynamics.

---

## Issue → Weak labeling

### Problem

Every track inside a violent video gets label `1`.

This includes:

- bystanders
- spectators
- unrelated people

---

### Fix

Prefer:

- pairwise interaction samples
- active participants
- tracks near conflict regions

---

### Why this matters

Otherwise the model learns:

```text
violent scene context
```

instead of actual fighting behavior.

---

## Issue → Missing-joint handling absent

### Problem

Missing or weak joints are treated as valid coordinates.

---

### Fix

Use:

- missing masks
- zero-filling
- confidence-aware features

---

### Why this matters

Occlusion is extremely common during fights.

The model must know the difference between:

```text
missing joint
vs
real movement
```

---

# 3) train.py

---

## Issue → Model heavily biased toward positive predictions

### Problem

Both of these exist:

```python
class_weights[1] *= 3.0
```

and:

```python
threshold = 0.3
```

This aggressively pushes the model toward:

```text
predicting fight
```

---

### Fix

Remove:

```python
class_weights[1] *= 3.0
```

Use only balanced weights initially.

---

### Why this matters

The current setup rewards false positives heavily.

The model learns:

```text
better safe than sorry
```

which becomes:

```text
people standing close = fight
```

---

## Issue → Hardcoded threshold

### Problem

`0.3` is extremely low.

---

### Fix

Tune threshold using validation data.

Example:

```python
for t in np.linspace(0.1, 0.9, 81):
```

selecting the threshold with best:

- F1
- precision/recall balance

---

### Why this matters

Fight detection systems need controlled false positives.

Otherwise alerts become unusable.

---

## Issue → Current architecture is not a true TCN

### Problem

Current architecture:

```python
Conv1D
MaxPooling1D
```

is closer to a simple temporal CNN.

Pooling removes short motion bursts.

---

### Fix

Use:

- dilated convolutions
- residual blocks
- no aggressive pooling

---

### Why this matters

Fight motion is bursty:

- punches
- kicks
- shoves

Pooling can erase these events.

Dilated TCNs preserve temporal detail.

---

## Issue → Evaluation focuses only on loss

### Problem

Training mainly monitors:

```python
val_loss
```

---

### Fix

Track:

- precision
- recall
- F1
- false positive rate

---

### Why this matters

Low loss does NOT guarantee usable alerts.

False positives are critical in CCTV systems.

---

## Issue → Prediction shape cleanup

### Problem

`y_pred_prob` has shape:

```text
(n, 1)
```

---

### Fix

Flatten predictions:

```python
y_pred = (y_pred_prob.ravel() > threshold).astype(int)
```

---

### Why this matters

Avoids shape inconsistencies during evaluation.

---

## Issue → Batch size too large

### Problem

```python
BATCH_SIZE = 1024
```

can hurt generalization.

---

### Fix

Try:

```python
64
128
256
```

---

### Why this matters

Smaller batches often improve learning diversity for temporal behavior tasks.

---

## Issue → cache() may become problematic

### Problem

`cache()` stores the entire dataset in memory.

---

### Fix

Only use if dataset comfortably fits RAM.

---

### Why this matters

Large datasets may cause memory pressure.

---

## Issue → No seed control

### Problem

Training results vary between runs.

---

### Fix

Set seeds for:

- TensorFlow
- NumPy
- Python random

---

### Why this matters

Makes experiments reproducible.

---

# Recommended Final Pipeline

```text
YOLOv8-pose
    ↓
stable tracking
    ↓
save frame-level + track-level data
    ↓
normalize skeletons
    ↓
compute velocity features
    ↓
build pairwise interaction sequences
    ↓
residual dilated TCN
    ↓
validation-based threshold tuning
    ↓
fight probability
```

---

# Highest-Impact Fixes

If only a few things are changed, prioritize these:

1. Remove:

```python
class_weights[1] *= 3.0
```

2. Stop using:

```python
threshold = 0.3
```

3. Normalize skeleton coordinates

4. Add velocity features

5. Use pairwise interaction sequences instead of single-person sequences

These changes alone should significantly reduce:

```text
standing close together → predicted as fighting
```
