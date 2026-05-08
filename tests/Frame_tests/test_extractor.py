import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data.frame_extraction import extract_frames

video_path = "dataset/Violence/V_100.mp4"


print("Exists:", os.path.exists(video_path))
print("Absolute Path:", os.path.abspath(video_path))

frames = extract_frames(video_path)

print(frames.shape)