"""
Manual contrast stretching pipeline for videos before YOLO detection/tracking.

Goal:
- Enhance frame contrast manually (no high-level auto enhancement functions)
- Stretch pixel intensity range per frame
- Process videos frame-by-frame for detection pipelines

Pipeline:
Video -> Read Frame -> Compute Min/Max -> Normalize -> Stretch -> Save

Author: Taher
"""

import os
import cv2
import argparse
import numpy as np


# =========================================================
# CONFIG
# =========================================================

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]


# =========================================================
# MANUAL CONTRAST STRETCHING
# =========================================================

def manual_contrast_stretch(frame):
    """
    Apply contrast stretching manually per channel.

    Formula:
        new = (pixel - min) * (255 / (max - min))

    No histogram equalization or built-in enhancement used.
    """

    # Convert to float for safe math operations
    h = frame.shape[0]
    w = frame.shape[1]

    stretched = np.zeros((h, w, 3), dtype=np.uint8)

    # Process each channel manually
    c = 0
    while c < 3:

        channel = frame[:, :, c]

        # manual min/max (no np.min / np.max shortcuts if strictly "low-level" required)
        min_val = 255
        max_val = 0

        i = 0
        while i < h:
            j = 0
            while j < w:
                val = channel[i][j]

                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val

                j += 1
            i += 1

        # avoid division by zero
        if max_val == min_val:
            c += 1
            continue

        scale = 255.0 / (max_val - min_val)

        # apply transformation manually
        i = 0
        while i < h:
            j = 0
            while j < w:
                val = channel[i][j]

                new_val = int((val - min_val) * scale)

                if new_val < 0:
                    new_val = 0
                elif new_val > 255:
                    new_val = 255

                stretched[i][j][c] = new_val
                j += 1

            i += 1

        c += 1

    return stretched


# =========================================================
# VIDEO PROCESSING
# =========================================================

def process_video(input_path, output_path, overwrite=False):
    """
    Apply contrast stretching frame-by-frame.
    """

    if os.path.exists(output_path) and not overwrite:
        print(f"[SKIPPED] {output_path}")
        return

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = os.path.dirname(output_path)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"[ERROR] Cannot write {output_path}")
        cap.release()
        return

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame is None:
            continue

        # ============================
        # CONTRAST STRETCHING STEP
        # ============================
        processed = manual_contrast_stretch(frame)

        writer.write(processed)

        frame_index += 1

        print(
            f"\rProcessing {os.path.basename(input_path)} "
            f"[{frame_index}/{frame_count}]",
            end=""
        )

    print()

    cap.release()
    writer.release()

    print(f"[DONE] {output_path}")


# =========================================================
# DIRECTORY PROCESSING
# =========================================================

def process_directory(input_dir, output_dir, overwrite=False):
    """
    Process all videos in a folder.
    """

    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)

    video_files = []

    # manual filtering
    for f in files:
        name = f.lower()

        for ext in VIDEO_EXTENSIONS:
            if name.endswith(ext):
                video_files.append(f)
                break

    video_files.sort()

    total = len(video_files)

    i = 0
    while i < total:

        file_name = video_files[i]

        print()
        print(f"========== VIDEO {i+1}/{total} ==========")
        print(file_name)

        in_path = os.path.join(input_dir, file_name)
        out_path = os.path.join(output_dir, file_name)

        process_video(in_path, out_path, overwrite)

        i += 1


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-root", type=str, default="data/videos")
    parser.add_argument("--output-root", type=str, default="data/videos_contrast")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    categories = ["Violence", "NonViolence"]

    for category in categories:

        print()
        print("=================================================")
        print("CATEGORY:", category)
        print("=================================================")

        in_dir = os.path.join(args.input_root, category)
        out_dir = os.path.join(args.output_root, category)

        process_directory(in_dir, out_dir, args.overwrite)

    print()
    print("====================================")
    print("CONTRAST STRETCHING COMPLETE")
    print("====================================")