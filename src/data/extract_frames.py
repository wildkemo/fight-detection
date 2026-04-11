import cv2
import os
import numpy as np

def extract_frames(video_path, target_fps=5):
    """
    Extracts frames from a video perfectly evenly distributed across the 
    entire video duration. This ensures the last frame is captured and 
    no parts of the video are missed.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Total frames and FPS from video metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0 or original_fps <= 0:
        cap.release()
        raise ValueError(f"Invalid metadata for video: {video_path}")

    # Calculate target frame count based on duration
    duration = total_frames / original_fps
    target_count = int(duration * target_fps)
    
    if target_count <= 0:
        cap.release()
        return []

    # Generate evenly spaced indices from 0 to (total_frames - 1)
    # Using np.linspace guarantees the last frame is included.
    indices = np.linspace(0, total_frames - 1, target_count, dtype=int)
    indices_set = set(indices)
    
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count in indices_set:
            frames.append(frame)
        count += 1

    cap.release()
    return frames
