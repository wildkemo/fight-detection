"""
Manual letterbox resize pipeline for videos before YOLO detection/tracking.

Goal:
- Resize frames while preserving aspect ratio
- Add padding manually (letterbox)
- Avoid high-level helper abstractions as much as possible
- Produce fixed-size frames for YOLO models

Pipeline:
Video -> Read Frame -> Compute Scale -> Resize -> Create Canvas -> Paste -> Save

Author: Taher
"""

import os
import cv2
import argparse
import numpy as np


# =========================================================
# CONFIG
# =========================================================

TARGET_WIDTH = 640
TARGET_HEIGHT = 640

PADDING_COLOR = (114, 114, 114)

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]


# =========================================================
# MANUAL LETTERBOX FUNCTION
# =========================================================

def manual_letterbox(frame, target_width, target_height):
    """
    Resize image while preserving aspect ratio,
    then manually add padding.

    Returns:
        padded_frame
    """

    # ---------------------------------------------
    # ORIGINAL FRAME SIZE
    # ---------------------------------------------
    original_height = frame.shape[0]
    original_width = frame.shape[1]

    # ---------------------------------------------
    # COMPUTE SCALE FACTOR
    # ---------------------------------------------
    width_scale = target_width / original_width
    height_scale = target_height / original_height

    # Use smaller scale to preserve aspect ratio
    if width_scale < height_scale:
        scale = width_scale
    else:
        scale = height_scale

    # ---------------------------------------------
    # NEW RESIZED DIMENSIONS
    # ---------------------------------------------
    resized_width = int(original_width * scale)
    resized_height = int(original_height * scale)

    # ---------------------------------------------
    # RESIZE FRAME
    # ---------------------------------------------
    resized_frame = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR
    )

    # ---------------------------------------------
    # CREATE EMPTY CANVAS
    # ---------------------------------------------
    canvas = np.full(
        (
            target_height,
            target_width,
            3
        ),
        PADDING_COLOR,
        dtype=np.uint8
    )

    # ---------------------------------------------
    # COMPUTE PADDING
    # ---------------------------------------------
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2

    # ---------------------------------------------
    # PLACE IMAGE INTO CENTER OF CANVAS
    # ---------------------------------------------
    canvas[
        pad_y:pad_y + resized_height,
        pad_x:pad_x + resized_width
    ] = resized_frame

    return canvas


# =========================================================
# VIDEO RESIZE FUNCTION
# =========================================================

def resize_video(
    input_video_path,
    output_video_path,
    target_width,
    target_height,
    overwrite=False
):
    """
    Read video frame-by-frame,
    apply manual letterbox resize,
    and save output video.
    """

    # ---------------------------------------------
    # SKIP IF EXISTS
    # ---------------------------------------------
    if os.path.exists(output_video_path) and not overwrite:
        print(f"[SKIPPED] {output_video_path}")
        return

    # ---------------------------------------------
    # OPEN VIDEO
    # ---------------------------------------------
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video:")
        print(input_video_path)
        return

    # ---------------------------------------------
    # VIDEO INFO
    # ---------------------------------------------
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # ---------------------------------------------
    output_dir = os.path.dirname(output_video_path)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------
    # VIDEO WRITER
    # ---------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (target_width, target_height)
    )

    if not writer.isOpened():
        print(f"[ERROR] Cannot create output video:")
        print(output_video_path)

        cap.release()
        return

    # ---------------------------------------------
    # PROCESS FRAMES
    # ---------------------------------------------
    current_frame = 0

    while True:

        success, frame = cap.read()

        # End of video
        if not success:
            break

        # Invalid frame
        if frame is None:
            continue

        # -----------------------------------------
        # LETTERBOX RESIZE
        # -----------------------------------------
        output_frame = manual_letterbox(
            frame,
            target_width,
            target_height
        )

        # -----------------------------------------
        # WRITE FRAME
        # -----------------------------------------
        writer.write(output_frame)

        current_frame += 1

        # -----------------------------------------
        # SIMPLE PROGRESS
        # -----------------------------------------
        print(
            f"\rProcessing: {os.path.basename(input_video_path)} "
            f"[{current_frame}/{frame_count}]",
            end=""
        )

    print()

    # ---------------------------------------------
    # RELEASE
    # ---------------------------------------------
    cap.release()
    writer.release()

    print(f"[DONE] {output_video_path}")


# =========================================================
# DIRECTORY PROCESSING
# =========================================================

def process_directory(
    input_directory,
    output_directory,
    target_width,
    target_height,
    limit=None,
    overwrite=False
):
    """
    Resize all videos inside a directory.
    """

    # ---------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # ---------------------------------------------
    os.makedirs(output_directory, exist_ok=True)

    # ---------------------------------------------
    # GET VIDEO FILES
    # ---------------------------------------------
    all_files = os.listdir(input_directory)

    video_files = []

    for file_name in all_files:

        lower_name = file_name.lower()

        for extension in VIDEO_EXTENSIONS:

            if lower_name.endswith(extension):
                video_files.append(file_name)
                break

    # ---------------------------------------------
    # SORT FILES
    # ---------------------------------------------
    video_files.sort()

    # ---------------------------------------------
    # LIMIT FILES
    # ---------------------------------------------
    if limit is not None:
        video_files = video_files[:limit]

    # ---------------------------------------------
    # PROCESS EACH VIDEO
    # ---------------------------------------------
    total_videos = len(video_files)

    for index in range(total_videos):

        file_name = video_files[index]

        print()
        print(f"========== VIDEO {index + 1}/{total_videos} ==========")
        print(file_name)

        input_path = os.path.join(input_directory, file_name)

        output_path = os.path.join(output_directory, file_name)

        resize_video(
            input_path,
            output_path,
            target_width,
            target_height,
            overwrite
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Target output width"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Target output height"
    )

    parser.add_argument(
        "--input-root",
        type=str,
        default="data/videos",
        help="Input dataset root"
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="data/videos_letterboxed",
        help="Output dataset root"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    # =====================================================
    # PROCESS DATASET CATEGORIES
    # =====================================================

    categories = [
        "Violence",
        "NonViolence"
    ]

    for category in categories:

        print()
        print("=================================================")
        print(f"PROCESSING CATEGORY: {category}")
        print("=================================================")

        input_dir = os.path.join(
            args.input_root,
            category
        )

        output_dir = os.path.join(
            args.output_root,
            category
        )

        process_directory(
            input_dir,
            output_dir,
            args.width,
            args.height,
            args.limit,
            args.overwrite
        )

    print()
    print("====================================")
    print("LETTERBOX VIDEO RESIZE COMPLETE")
    print("====================================")