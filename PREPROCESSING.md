# ⚙️ Video Preprocessing Pipeline

## Goal

Enhance the visual clarity of subjects (people) and emphasize structural details (edges) within CCTV footage using an optimized, grayscale-only pipeline designed for maximum processing speed and analytical precision.

---

## Step 1: Extract Frames
**Implementation:** `extract_frames.py`
**What:** Sample frames from video files at a consistent rate (default: 5.0 FPS).
**Why:** Converts temporal data into discrete spatial snapshots while balancing detail retention with data volume.

---

## Phase 1: Grayscale Conversion
**Implementation:** `color_space.py`
**What:** Convert BGR frames to Grayscale immediately after extraction.
**Why:** Standardizing to 1-channel data early in the pipeline significantly reduces computational overhead for all subsequent filters (Denoising, Sharpening, etc.), maximizing processing speed.

---

## Step 2: Resize and Standardize (Skipped)
**Implementation:** `resize.py`
**What:** Resize frames to uniform dimensions (e.g., 224x224).
**Why:** Currently skipped to preserve original resolution for superior edge analysis.

---

## Step 3: Dual-Stage Denoising
**Implementation:** `denoise.py`
**What:** A sequential application of a $3 \times 3$ **Median Filter** followed by a **Guided Filter** ($r=1, \epsilon=0.01$).
**Why:**
- **Median Filter:** Specifically targets salt-and-pepper sensor noise common in low-light CCTV.
- **Guided Filter:** Performs edge-preserving smoothing, removing remaining artifacts while ensuring critical boundaries stay sharp.

---

## Step 4: Contrast Stretching (Normalization)
**Implementation:** `dynamic_range.py`
**What:** Linearly scales intensity levels to occupy the full $[0, 255]$ range.
**Why:** Ensures that the frame utilizes the maximum available dynamic range, making subtle variations more visible before non-linear transformations.

---

## Step 10 & 11: Adaptive Dynamic Range Adjustment
**Implementation:** `log_transform.py` & `inverse_log.py`
**What:** Conditionally applies a **Log Transformation** (for dark frames) or an **Inverse Log/Power Transformation** (for bright frames) based on the frame's mean brightness.
**Why:**
- **Log:** Expands intensity values in dark regions (shadows).
- **Inverse Log:** Expands intensity values in bright regions (highlights).
- **Adaptive Logic:** Prevents "washing out" the image by only applying the curve that helps correct the specific lighting imbalance.

---

## Step 5: Global Histogram Equalization
**Implementation:** `contrast.py`
**What:** Manual implementation of global histogram equalization using optimized Look-Up Tables (LUT).
**Why:** Spreads out the most frequent intensity values, increasing global contrast and making human subjects stand out from backgrounds.

---

## Step 6: Unsharp Masking (Sharpening)
**Implementation:** `sharpen.py`
**What:** Extracts high-frequency details (edges) and adds them back to the original image ($Amount=5.0$).
**Why:** Unlike standard kernels, unsharp masking enhances edges and subtle differences proportionally, creating a crisp analytical look that highlights motion boundaries.

---

## Step 8: Sanity Check
**Implementation:** `sanity_check.py`
**What:** Programmatic inspection of processed frames to ensure sufficient variance.
**Why:** Ensures that enhancement steps haven't introduced artifacts or resulted in low-variance (solid block) data.

---

## Step 9: Organized Storage
**Implementation:** `storage.py`
**What:** Saves frames into a structured hierarchy at the project root: `output/<category>/<video_name>/burst_<index>/frame_<index>.jpg`.
**Why:** Isolates data by category (Violence/NonViolence) and groups frames into bursts for easier sequence analysis.

---

## Key Principle
The pipeline prioritizes a **Grayscale-First** workflow to achieve maximum throughput while utilizing **Edge-Preserving** denoising and **Adaptive** lighting corrections to maximize the visibility of human interactions.
