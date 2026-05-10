# Evaluation Strategy — Multi-Model Fight Detection System

Your system has 3 stages:
YOLO → MoveNet → GRU → Final decision

So evaluation must happen in **layers**, not just one place.

---

# 1. Component-Level Evaluation (Debug Each Model)

## 🟡 YOLO (Person Detection)

### Goal

Check if people are correctly detected in frames.

### Metrics

- Precision
- Recall
- Miss rate (VERY important)

### Why it matters

If YOLO misses a person → everything downstream breaks.

### What you test

- crowded scenes
- low light
- motion blur

---

## 🟡 MoveNet (Pose Estimation)

### Goal

Check quality of extracted skeletons.

### What you evaluate

- missing keypoints rate
- unstable joints (jitter)
- confidence scores

### Why it matters

Bad pose = garbage input to GRU

---

## 🟡 GRU (Temporal Classifier)

### Goal

Evaluate motion understanding.

### Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Important

This is evaluated on:
👉 **pose sequences, not images**

---

# 2. End-to-End Evaluation (Most Important)

## Goal

Test full pipeline:

```text
video → prediction (fight / non-fight)
```

---

## What you evaluate

### ✔ Video-level accuracy

Each video has ONE label.

Example:

| Video | Ground Truth | Prediction |
| ----- | ------------ | ---------- |
| v1    | fight        | fight      |
| v2    | nonfight     | fight ❌   |

---

## Metrics

- Accuracy
- Precision (important for false alarms)
- Recall (important for missing fights)
- F1-score

---

## Why video-level matters

Because your system outputs:

```text
sequence → single decision
```

not per-frame predictions.

---

# 3. Real-Time Evaluation (Deployment Test)

## Goal

Simulate real CCTV usage.

### Input

- laptop webcam OR recorded video stream

---

## What you measure

### Performance

- FPS (frames per second)
- latency per inference

### Reliability

- false positives per hour
- missed fight events

### Stability

- prediction consistency over time

---

# 4. Recommended Evaluation Workflow

## Step 1 — GRU-only evaluation

Test model in isolation:

```text
pose sequences → GRU → prediction
```

Check:

- confusion matrix
- F1 score

---

## Step 2 — Full pipeline evaluation (offline videos)

Run full system:

```text
video → YOLO → MoveNet → GRU → output
```

Compare with labels.

---

## Step 3 — Real-time testing

Use webcam:

```text
live stream → prediction stream
```

Measure:

- stability
- lag
- false alerts

---

# 5. Key Insight

You do NOT evaluate everything at once initially.

You debug step-by-step:

```text
1. YOLO correct?
2. MoveNet stable?
3. GRU accurate?
4. Full system reliable?
```

---

# Final Summary

You evaluate in TWO layers:

## ✔ Component-level

- YOLO → detection quality
- MoveNet → pose quality
- GRU → sequence classification

## ✔ System-level

- full video prediction accuracy
- real-time performance metrics

This is the correct way to evaluate multi-stage AI systems.

```

```
