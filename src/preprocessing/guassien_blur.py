"""
Manual Gaussian blur pipeline for videos before detection/tracking.

Goal:
- Smooth frames with a Gaussian filter to reduce sensor noise
- Build the kernel and convolve manually (no cv2.GaussianBlur)
- Process videos frame-by-frame in the same style as resize.py

Pipeline:
Video -> Read Frame -> Build Kernel -> Horizontal Pass -> Vertical Pass -> Save
"""

import os
import cv2
import math
import argparse
import numpy as np


# =========================================================
# CONFIG
# =========================================================

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

KERNEL_SIZE = 5
SIGMA = 1.0

MIN_OUTPUT = 0
MAX_OUTPUT = 255


# =========================================================
# MANUAL GAUSSIAN BLUR HELPERS
# =========================================================

def manual_gaussian_weight(offset, sigma):
    """
    Compute one sample of a 1D Gaussian function.
    """
    numerator = -(float(offset) * float(offset))
    denominator = 2.0 * float(sigma) * float(sigma)
    return math.exp(numerator / denominator)


def manual_build_gaussian_kernel(kernel_size, sigma):
    """
    Build and normalize a 1D Gaussian kernel manually.
    """
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1

    center_index = kernel_size // 2
    raw_kernel = []
    kernel_sum = 0.0

    for kernel_index in range(kernel_size):
        offset = kernel_index - center_index
        weight = manual_gaussian_weight(offset, sigma)
        raw_kernel.append(weight)
        kernel_sum += weight

    if kernel_sum <= 0.0:
        return [1.0 / float(kernel_size)] * kernel_size

    return [w / kernel_sum for w in raw_kernel]


def manual_convolve_horizontal(channel, kernel):
    """
    Apply a 1D horizontal convolution using numpy vectorization.
    """
    image_height, image_width = channel.shape
    kernel_radius = len(kernel) // 2

    # Pad the channel manually to simulate border replication
    padded = np.empty((image_height, image_width + 2 * kernel_radius), dtype=np.float32)
    padded[:, kernel_radius:-kernel_radius] = channel
    
    # Fill padding with edge values
    for i in range(kernel_radius):
        padded[:, i] = channel[:, 0]
        padded[:, -1 - i] = channel[:, -1]

    # Vectorized accumulation across the entire image
    output_channel = np.zeros_like(channel, dtype=np.float32)
    for k_idx, k_val in enumerate(kernel):
        # Slice and multiply
        output_channel += padded[:, k_idx : k_idx + image_width] * k_val

    # Clip and round
    return np.clip(np.round(output_channel), MIN_OUTPUT, MAX_OUTPUT).astype(np.uint8)


def manual_convolve_vertical(channel, kernel):
    """
    Apply a 1D vertical convolution using numpy vectorization.
    """
    image_height, image_width = channel.shape
    kernel_radius = len(kernel) // 2

    # Pad the channel manually to simulate border replication
    padded = np.empty((image_height + 2 * kernel_radius, image_width), dtype=np.float32)
    padded[kernel_radius:-kernel_radius, :] = channel
    
    # Fill padding with edge values
    for i in range(kernel_radius):
        padded[i, :] = channel[0, :]
        padded[-1 - i, :] = channel[-1, :]

    # Vectorized accumulation across the entire image
    output_channel = np.zeros_like(channel, dtype=np.float32)
    for k_idx, k_val in enumerate(kernel):
        # Slice and multiply
        output_channel += padded[k_idx : k_idx + image_height, :] * k_val

    # Clip and round
    return np.clip(np.round(output_channel), MIN_OUTPUT, MAX_OUTPUT).astype(np.uint8)


def manual_blur_channel(channel, kernel_size, sigma):
    """
    Separable Gaussian blur on one channel:
    horizontal pass, then vertical pass.
    """
    kernel = manual_build_gaussian_kernel(kernel_size, sigma)
    horizontal_result = manual_convolve_horizontal(channel, kernel)
    vertical_result = manual_convolve_vertical(horizontal_result, kernel)
    return vertical_result


def manual_gaussian_blur(frame, kernel_size, sigma):
    """
    Apply manual Gaussian blur to each BGR channel.
    """
    if frame is None:
        return None

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        print("[ERROR] Expected a 3-channel BGR frame.")
        return frame

    blurred_blue = manual_blur_channel(frame[:, :, 0], kernel_size, sigma)
    blurred_green = manual_blur_channel(frame[:, :, 1], kernel_size, sigma)
    blurred_red = manual_blur_channel(frame[:, :, 2], kernel_size, sigma)

    output_frame = np.empty_like(frame)
    output_frame[:, :, 0] = blurred_blue
    output_frame[:, :, 1] = blurred_green
    output_frame[:, :, 2] = blurred_red

    return output_frame


# =========================================================
# VIDEO PROCESSING
# =========================================================

def gaussian_blur_video(
    input_video_path,
    output_video_path,
    kernel_size,
    sigma,
    overwrite=False
):
    """
    Read a video frame-by-frame, apply manual Gaussian blur, and save output.
    """

    if os.path.exists(output_video_path) and not overwrite:
        print(f"[SKIPPED] {output_video_path}")
        return

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = os.path.dirname(output_video_path)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    if not writer.isOpened():
        print(f"[ERROR] Cannot create output video: {output_video_path}")
        cap.release()
        return

    current_frame = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame is None:
            continue

        output_frame = manual_gaussian_blur(
            frame,
            kernel_size,
            sigma
        )

        if output_frame is None:
            continue

        writer.write(output_frame)
        current_frame += 1

        print(
            f"\rProcessing: {os.path.basename(input_video_path)} "
            f"[{current_frame}/{frame_count}]",
            end=""
        )

    print()
    cap.release()
    writer.release()
    print(f"[DONE] {output_video_path}")


# =========================================================
# DIRECTORY PROCESSING
# =========================================================

def process_directory(
    input_directory,
    output_directory,
    kernel_size,
    sigma,
    limit=None,
    overwrite=False
):
    """
    Apply Gaussian blur to all videos inside a directory.
    """
    os.makedirs(output_directory, exist_ok=True)
    all_files = os.listdir(input_directory)
    video_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)]
    video_files.sort()

    if limit is not None:
        video_files = video_files[:limit]

    total_videos = len(video_files)

    for index, file_name in enumerate(video_files):
        print(f"\n========== VIDEO {index + 1}/{total_videos} ==========\n{file_name}")
        input_path = os.path.join(input_directory, file_name)
        output_path = os.path.join(output_directory, file_name)

        gaussian_blur_video(
            input_path,
            output_path,
            kernel_size,
            sigma,
            overwrite
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Gaussian blur for surveillance videos.")
    parser.add_argument("--input-root", type=str, default="data/videos_clahe")
    parser.add_argument("--output-root", type=str, default="data/videos_gaussian_blur")
    parser.add_argument("--kernel-size", type=int, default=KERNEL_SIZE)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    categories = ["Violence", "NonViolence"]

    for category in categories:
        print(f"\n=================================================\nPROCESSING CATEGORY: {category}\n=================================================")
        input_dir = os.path.join(args.input_root, category)
        output_dir = os.path.join(args.output_root, category)

        if not os.path.isdir(input_dir):
            print(f"[WARNING] Input directory not found: {input_dir}")
            continue

        process_directory(
            input_dir,
            output_dir,
            args.kernel_size,
            args.sigma,
            args.limit,
            args.overwrite
        )

    print("\n====================================\nGAUSSIAN BLUR PREPROCESSING COMPLETE\n====================================")