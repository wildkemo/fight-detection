# ⚙️ Video Preprocessing Pipeline

## Goal

Prepare videos in a consistent way that helps the model learn useful patterns, not just “clear” images.

---

## Step 1: Split Dataset by Video

**What:**
Split videos into train/test sets

**Why:**
Avoid data leakage (same video in both sets)

---

## Step 2: Extract Frames

**What:**
Sample frames at 5–10 FPS

**Why:**

- Reduce redundancy
- Capture motion effectively

---

## Step 3: Resize Frames

**What:**
Resize all frames (e.g., 224×224)

**Why:**

- Required for models
- Reduces computation

---

## Step 4: Normalize Pixel Values

**What:**
Scale pixels from [0–255] to [0–1]

**Why:**

- Stabilizes training
- Improves convergence

---

## Step 5: Color Processing

**What:**

- Keep RGB OR convert to grayscale

**Why:**

- RGB = more info
- Grayscale = simpler, less noise

---

## Step 6: Light Denoising

**What:**
Apply slight smoothing

**Why:**
Remove noise without losing motion details

---

## Step 7: Contrast Normalization

**What:**
Apply histogram equalization or similar

**Why:**
Handle lighting differences across videos

---

## Step 8: Sequence Construction

**What:**
Group frames into sequences (e.g., 16–30 frames)

**Why:**
Capture temporal information (motion)

---

## Step 9: Label Assignment

**What:**
Assign labels based on the video

**Why:**
Keep consistency between data and labels

---

## Step 10: Save Processed Data

**What:**
Organize into folders:

dataset/
train/
violence/
non_violence/
test/
violence/
non_violence/

**Why:**
Simplifies training pipeline

---

## Step 11: Sanity Check

**What:**
Visually inspect processed data

**Why:**
Ensure no useful info was lost

---

## Key Principle

Preprocessing should improve learning signal, not just visual quality.
