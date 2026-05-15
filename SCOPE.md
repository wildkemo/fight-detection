# 🚨 CPU-Optimized Fight Detection Pipeline (YOLOv8n + RTMPose)

This system is split into 5 small Python files. Each file does ONE clean job so you can debug, optimize, or replace parts independently without breaking everything.

The whole idea is: detection → tracking → pose → feature engineering → temporal model → decision.

---

# 📁 Project Structure

project/
│
├── src/
│ ├── detect_and_track.py
│ ├── extract_pose.py
│ ├── build_sequences.py
│ ├── train_tcn.py
│ └── inference.py
│
├── models/
│ ├── yolov8n.onnx
│ ├── rtmpose.onnx
│ └── tcn_model.tflite
│
├── data/
│ ├── tracks/
│ ├── poses/
│ ├── sequences/
│ └── videos/
│   ├── Violence/
│   └── NonViolence/

---

# 1) detect_and_track.py

## What this file does

This is the “who is in the scene and where are they?” stage.

It ONLY:

- detects people in each frame using YOLOv8n
- assigns consistent IDs across frames using a tracker
- stores bounding boxes over time

It does NOT do pose estimation, ML classification, or any temporal logic.

---

## How it works step-by-step

1. Read video stream from /data/videos (sample at 12-fps)
2. Run YOLOv8n (person detection only, NOT pose)
3. Pass detections into a tracker (ByteTrack or BoT-SORT)
4. Tracker assigns stable `track_id` per person
5. Save per-frame results for each video (categorized by violence and nonviolence in separate folders)

---

## Output

Each frame becomes:

{
"frame_id": 15,
"track_id": 3,
"bbox": [x1, y1, x2, y2],
"confidence": 0.91
}

---

## Why this exists

This separates:

- detection problems (YOLO)
- identity problems (tracking)

So later stages don’t deal with raw video complexity.

---

# 2) extract_pose.py

## What this file does

This stage converts tracked humans into skeletons.

It takes:

- bounding boxes from tracking
  and produces:
- human keypoints using RTMPose

---

## How it works step-by-step

1. Load tracking output
2. For each frame:
   - crop each person using bbox
3. Run RTMPose on each crop
4. Extract 17 keypoints per person
5. Convert crop coordinates → full image coordinates
6. Save per track over time

---

## Output

{
"track_id": 3,
"frame_id": 15,
"keypoints": [[x,y,conf], ...]
}

---

## Why RTMPose here

RTMPose is used instead of YOLO-pose because:

- more stable keypoints
- less jitter across frames
- better under occlusion
- better for temporal models

---

## Important detail

RTMPose outputs are relative to the crop, so you MUST convert:

crop coords → full-frame coords

Otherwise sequences become inconsistent.

---

# 3) build_sequences.py (UPGRADED VERSION)

## What this file does

This is the most important preprocessing stage in the entire system.

It converts raw RTMPose skeleton tracks into **high-quality temporal interaction features** that a TCN can actually learn from.

Instead of just feeding coordinates, this version builds a **motion + interaction + energy representation** that is much more robust for fight detection.

---

## How it works step-by-step

---

## Step 1: Load pose data

Reads structured pose outputs:

data/poses/\*.json

Each file contains:

- multiple track_ids
- per-frame keypoints
- time-ordered skeleton sequences

Each track represents one person over time.

---

## Step 2: Skeleton normalization

Raw keypoints depend heavily on:

- camera position
- zoom level
- person distance from camera

So we normalize to make everything consistent:

kpts = kpts - pelvis_center
kpts = kpts / torso_length

### Why this matters

This makes features:

- scale invariant
- translation invariant
- camera independent

So the model focuses on motion, not position.

---

## Step 3: Basic motion features (core temporal signal)

Instead of only using raw keypoints, we compute motion:

velocity:
v[t] = kpts[t] - kpts[t-1]

acceleration:
a[t] = v[t] - v[t-1]

### Why this matters

Fights are not static poses — they are:

- sudden bursts
- fast directional changes
- repeated impact motion

---

## Step 4: Motion energy features (NEW IMPORTANT ADDITION)

We compute overall movement intensity:

energy[t] = sum(||kpts[t] - kpts[t-1]||^2)

### Why this helps

- standing still → low energy
- walking → medium stable energy
- fighting → sharp spikes + instability

This is one of the strongest global indicators.

---

## Step 5: Joint-level interaction features (VERY IMPORTANT)

We compute distances between critical joints:

- wrist ↔ head
- wrist ↔ torso
- elbow ↔ head
- hand ↔ hand

Example:

d(hand, head), d(wrist, torso)

### Why this matters

Punching and aggressive gestures directly appear here.

---

## Step 6: Pairwise interaction features (CORE OF FIGHT DETECTION)

Instead of analyzing single people, we model relationships:

Person A ↔ Person B

We compute:

- distance between centers
- relative velocity
- closing speed
- direction alignment

relative velocity:
v_rel = v_A - v_B

### Why this is critical

Fights are not individual behaviors — they are **interactions**.

---

## Step 7: Angular motion features (NEW STRONG SIGNAL)

We compute joint angles:

- elbow angle
- shoulder angle
- torso twist

### Why this helps

Punches and grappling produce:

- fast angular changes
- sharp directional rotations

These are hard to fake in non-violent actions.

---

## Step 8: Motion asymmetry features (NEW)

We measure imbalance between two people:

asymmetry = |energy_A - energy_B|

### Why this matters

- normal interaction → symmetric motion
- fight → one dominates, one reacts violently

---

## Step 9: Temporal statistics (stability features)

We compute over sliding windows:

- variance of motion
- trend slope (increasing/decreasing activity)
- short-term volatility

### Why this matters

Fights often:

- escalate over time
- show unstable motion patterns

---

## Step 10: Build interaction sequences (36 frames)

We create fixed-length temporal windows:

36 frames → one training sample

Each sample contains:

(36, feature_dim)

Where feature_dim includes:

- normalized pose
- velocity
- acceleration
- energy
- angular motion
- pairwise interaction features
- asymmetry metrics
- temporal statistics

---

## Step 11: Dataset split (IMPORTANT ADDITION)

We explicitly create:

X_train.npy, y_train.npy  
X_val.npy, y_val.npy  
X_test.npy, y_test.npy

### Why validation is necessary

Without validation:

- thresholds are guessed
- smoothing is overfitted
- test results become misleading

Validation is used for:

- tuning TCN threshold
- adjusting smoothing window
- balancing class weights
- choosing interaction distance thresholds

---

## Step 12: Save final dataset

Outputs:

data/sequences/
├── X_train.npy
├── y_train.npy
├── X_val.npy
├── y_val.npy
├── X_test.npy
├── y_test.npy

---

## Why this file is the MOST important

This stage determines everything downstream.

If this is weak:

- model will always overfit
- false positives (standing = fight)
- unstable predictions

If this is strong:

- even a simple TCN performs well
- robust real-world detection
- stable CCTV behavior

---

# 4) train_tcn.py

## What this file does

This trains the temporal model that decides fight vs no fight.

---

## How it works step-by-step

1. Load numpy sequences
2. Build Temporal Convolutional Network
3. Train binary classifier
4. Evaluate performance
5. Export optimized model

---

## Model type

We use a Residual TCN:

- Dilated Conv1D layers (long-term motion)
- Residual connections (stable training)
- BatchNorm
- Dropout
- Dense classifier at end

---

## Output

tcn_model.tflite (INT8 quantized for CPU)

---

## Why TCN

Compared to LSTM/GRU:

- faster inference
- parallel processing
- better temporal pattern capture
- stable training

---

# 5) inference.py

## What this file does

This is the real-time system that runs on CCTV/live video.

It combines EVERYTHING:
detection + tracking + pose + features + model + decision.

---

## Full runtime pipeline

frame
→ YOLOv8n detection
→ ByteTrack / BoT-SORT tracking
→ interaction filtering (distance check)
→ RTMPose (ONLY nearby people)
→ skeleton normalization
→ motion feature computation
→ 36-frame sequence building
→ TFLite TCN inference
→ temporal smoothing
→ final fight decision

---

## Key optimization idea

DO NOT run pose on everyone.

Only run pose if:

if distance(personA, personB) < threshold:
run RTMPose

This is what makes the system CPU-feasible.

---

## Track memory structure

tracks[track_id] = {
"bbox": ...,
"buffer": deque(maxlen=36),
"motion": deque(maxlen=8),
"last_prob": ...
}

---

## Decision logic

We don’t trigger on a single frame.

Instead:

- TCN probability must be high
- AND it must persist over time

This prevents false positives like:

- standing close
- waving hands
- occlusions

---

## Optional recording

If fight detected:

- store pre-buffer (few seconds before event)
- start recording
- stop after cooldown

---

# 🚀 FINAL SYSTEM FLOW

Old system:
YOLOv8-pose → raw skeletons → simple TCN → unstable results

New system:
YOLOv8n → tracking → RTMPose → normalization → motion features → pairwise interaction modeling → residual TCN → temporal smoothing → robust detection

---

# 🧠 CORE IDEA SHIFT

Instead of:

❌ “this person is moving weird → fight”

We now do:

✅ “two people interacting with aggressive motion patterns → fight”

That is what actually works in real CCTV environments.
