import cv2
import os

def extract_frames(video_path, target_fps=5):
    """
    Extracts frames from a video at a specific frame rate.
    
    Args:
        video_path (str): Path to the video file.
        target_fps (int): Number of frames to extract per second.
        
    Yields:
        tuple: (frame_index, frame) where frame is a numpy array (BGR).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Error: Could not determine FPS for {video_path}")
        cap.release()
        return

    # Calculate interval between frames to keep
    frame_interval = max(1, int(round(fps / target_fps)))
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Validate frame: ensure it's not None or empty
        if frame is None or frame.size == 0:
            print(f"Warning: Decoded empty frame at index {frame_count} in {video_path}. Skipping.")
            frame_count += 1
            continue
            
        if frame_count % frame_interval == 0:
            yield saved_count, frame
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
