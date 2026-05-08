import sys
import os
import cv2
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.append(BASE_DIR)

from src.data.frame_extraction import extract_frames

video_path = os.path.join(
    BASE_DIR,
    "dataset",
    "Violence",
    "V_100.mp4"
)

print("Video exists:", os.path.exists(video_path))
print("Video path:", video_path)

frames = extract_frames(video_path)

if frames is None:
    print("Frame extraction failed ❌")
    exit()

print("Frames Shape:", frames.shape)

fig, axes = plt.subplots(4, 4, figsize=(12, 12))

num_frames = min(len(frames), 16)

for i in range(num_frames):
    axes.flat[i].imshow(frames[i])
    axes.flat[i].set_title(f"Frame {i+1}")
    axes.flat[i].axis("off")

for j in range(num_frames, 16):
    axes.flat[j].axis("off")

plt.tight_layout()
plt.show()