"""
Its job:

video frames → fixed sequence of frames

Example:

video_001 → 16 frames → label 1
video_002 → 16 frames → label 0

Why?
Because LSTM needs ordered frames, not random images.

"""

import os
import cv2
import numpy as np
from tqdm import tqdm

from src.utils.config import (
    PROCESSED_FRAMES_DIR,
    SEQUENCE_LENGTH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    CLASS_NAMES
)


def load_frame(frame_path):
    """
    Load and preprocess a single frame
    """
    img = cv2.imread(frame_path)

    if img is None:
        return None

    # Resize
    img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

    # Normalize
    img = img / 255.0

    return img


def build_sequence_from_video(video_path):
    """
    Build a fixed-length sequence from frames of one video
    """

    frames = sorted(os.listdir(video_path))

    sequence = []

    for frame_name in frames:
        frame_path = os.path.join(video_path, frame_name)

        img = load_frame(frame_path)

        if img is None:
            continue

        sequence.append(img)

    # If sequence is empty → skip
    if len(sequence) == 0:
        return None

    # If too short → pad
    if len(sequence) < SEQUENCE_LENGTH:
        last_frame = sequence[-1]

        while len(sequence) < SEQUENCE_LENGTH:
            sequence.append(last_frame)

    # If too long → sample uniformly
    if len(sequence) > SEQUENCE_LENGTH:
        indices = np.linspace(
            0, len(sequence) - 1, SEQUENCE_LENGTH
        ).astype(int)

        sequence = [sequence[i] for i in indices]

    return np.array(sequence)


def build_dataset():
    """
    Build full dataset (X, y)
    """

    X = []
    y = []

    for label, class_name in enumerate(CLASS_NAMES):

        class_path = os.path.join(PROCESSED_FRAMES_DIR, class_name)

        if not os.path.exists(class_path):
            print(f"Warning: {class_path} not found")
            continue

        videos = os.listdir(class_path)

        print(f"Processing class: {class_name}")

        for video_name in tqdm(videos):

            video_path = os.path.join(class_path, video_name)

            if not os.path.isdir(video_path):
                continue

            sequence = build_sequence_from_video(video_path)

            if sequence is None:
                continue

            X.append(sequence)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"\nFinal Dataset Shape:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    return X, y


if __name__ == "__main__":
    X, y = build_dataset()