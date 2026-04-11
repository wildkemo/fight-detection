# Implementation Plan: Fix Video Sampling Distribution

## Objective
Update the sampling logic in `src/data/extract_frames.py` to ensure frames are perfectly evenly distributed across the entire length of the video, preventing the issue where images from the end of the video are missed or under-represented.

## Scope & Impact
- Updates `src/data/extract_frames.py`
- Replaces the `frame_interval`-based modulo sampling with `numpy.linspace` to calculate exact, evenly-distributed frame indices across the entire video frame count.

## Proposed Solution

1. **Update `extract_frames.py`:**
    - Import `numpy`.
    - Retrieve `total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))`.
    - Retrieve `original_fps = cap.get(cv2.CAP_PROP_FPS)`.
    - Calculate the video duration: `duration = total_frames / original_fps`.
    - Calculate the target number of frames to extract: `target_count = int(duration * target_fps)`.
    - Generate perfectly distributed indices: `indices = np.linspace(0, total_frames - 1, target_count, dtype=int)`.
    - Convert `indices` to a set for fast lookup.
    - Loop through the video frames, keeping only the frames where the current `count` exists in the `indices` set.

This mathematical approach guarantees that the first frame (0) and the last frame (`total_frames - 1`) are always part of the sampled output (if `target_count > 1`), and all intermediate frames are exactly evenly spaced.

## Verification
- Run the full pipeline via `python -m src.data.pipeline`.
- Inspect the output in `tmp/sanity_check/violence/V_1/`.
- Confirm that the last frame of the original video is captured and the extracted frames are uniformly spaced.
