import cv2
import os

def extract_frames(video_path, fps=10):
    """
    Step 1: Extract frames from video files at a consistent rate.
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    hop = round(video_fps / fps)
    
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % hop == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames
