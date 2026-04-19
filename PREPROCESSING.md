# ⚙️ Video Preprocessing Pipeline

## Goal

Enhance the visual clarity of subjects (people) and emphasize structural details (edges) within CCTV footage to ensure high-quality visual data for analysis.

---

## Step 1: Extract Frames

**What:**
Sample frames from video files at a consistent rate (e.g., 5–10 FPS).

**Why:**

- Converts temporal data into discrete spatial snapshots.
- Balances detail retention with data volume.

---

## Step 2: Resize and Standardize (Skipped)

**What:**
Resize all frames to a uniform dimension.

**Why:**

- _Currently neglected to preserve original resolution for better edge analysis._

---

## Step 3: Denoising

**What:**
Apply spatial filters (e.g., Median Blur) to reduce sensor noise.

**Why:**

- Reduces sensor noise common in low-light CCTV footage.
- Prevents noise from being misidentified as edge detail.

---

## Step 4: Log transformation (Skipped)

**What:**
Apply a logarithmic or inverse-logarithmic transformation based on brightness.

**Why:**

- _Currently neglected to prioritize raw intensity values._

---

## Step 5: Contrast Enhancement (CLAHE)

**What:**
Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

**Why:**

- Enhances local contrast to make human figures stand out from the background without amplifying global noise.

---

## Step 6: Edge Enhancement (Guided Filter)

**What:**
Apply **Guided Image Filtering** to perform edge-preserving detail enhancement.

**Why:**

- **Halo Reduction:** Unlike Unsharp Masking, Guided Filters do not produce "ringing" or "halo" artifacts around high-contrast edges.
- **Noise Suppression:** It enhances structural details (like limbs and physical contact) while maintaining smoothness in non-edge regions.
- **Clarity:** Specifically makes the separation between moving subjects and static backgrounds more distinct.

---

## Step 7: Color Space Optimization (Grayscale)

**What:**
Convert all enhanced frames to Grayscale.

**Why:**

- Grayscale highlights intensity changes and edges, removing redundant color information.

---

## Step 8: Sanity Check

**What:**
Visually and programmatically inspect processed frames.

**Why:**

- Ensures that transformations haven't introduced artifacts or resulted in empty/low-variance data.

---

## Step 9: Organized Storage

**What:**
Save processed frames into a structured directory hierarchy: `output/<video_name>/burst_<index>/frame_<index>.jpg`.

**Why:**

- **Per-video:** Keeps data from different sources isolated.
- **Bursts:** Groups frames into 16-frame sequences for easier temporal analysis.

---

## Key Principle

Preprocessing must prioritize the visual separation of subjects from their environment and the sharpness of motion-defining edges.
