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

    normalized_kernel = []

    if kernel_sum <= 0.0:

        uniform_value = 1.0 / float(kernel_size)

        for kernel_index in range(kernel_size):
            normalized_kernel.append(uniform_value)

        return normalized_kernel

    for kernel_index in range(kernel_size):
        normalized_kernel.append(raw_kernel[kernel_index] / kernel_sum)

    return normalized_kernel


def manual_get_pixel_value(channel, row_index, col_index):
    """
    Read a pixel with edge clamping (border replication).
    """

    image_height = channel.shape[0]
    image_width = channel.shape[1]

    if row_index < 0:
        row_index = 0
    elif row_index >= image_height:
        row_index = image_height - 1

    if col_index < 0:
        col_index = 0
    elif col_index >= image_width:
        col_index = image_width - 1

    return float(channel[row_index, col_index])


def manual_convolve_horizontal(channel, kernel):
    """
    Apply a 1D horizontal convolution using manual pixel loops.
    """

    image_height = channel.shape[0]
    image_width = channel.shape[1]
    kernel_radius = len(kernel) // 2

    output_channel = np.zeros(
        (image_height, image_width),
        dtype=np.uint8
    )

    for row_index in range(image_height):

        for col_index in range(image_width):

            weighted_sum = 0.0

            for kernel_index in range(len(kernel)):

                sample_col = col_index + kernel_index - kernel_radius
                pixel_value = manual_get_pixel_value(
                    channel,
                    row_index,
                    sample_col
                )

                weighted_sum += pixel_value * kernel[kernel_index]

            blurred_value = int(weighted_sum + 0.5)

            if blurred_value < MIN_OUTPUT:
                blurred_value = MIN_OUTPUT
            elif blurred_value > MAX_OUTPUT:
                blurred_value = MAX_OUTPUT

            output_channel[row_index, col_index] = blurred_value

    return output_channel


def manual_convolve_vertical(channel, kernel):
    """
    Apply a 1D vertical convolution using manual pixel loops.
    """

    image_height = channel.shape[0]
    image_width = channel.shape[1]
    kernel_radius = len(kernel) // 2

    output_channel = np.zeros(
        (image_height, image_width),
        dtype=np.uint8
    )

    for row_index in range(image_height):

        for col_index in range(image_width):

            weighted_sum = 0.0

            for kernel_index in range(len(kernel)):

                sample_row = row_index + kernel_index - kernel_radius
                pixel_value = manual_get_pixel_value(
                    channel,
                    sample_row,
                    col_index
                )

                weighted_sum += pixel_value * kernel[kernel_index]

            blurred_value = int(weighted_sum + 0.5)

            if blurred_value < MIN_OUTPUT:
                blurred_value = MIN_OUTPUT
            elif blurred_value > MAX_OUTPUT:
                blurred_value = MAX_OUTPUT

            output_channel[row_index, col_index] = blurred_value

    return output_channel


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

    blue_channel = frame[:, :, 0]
    green_channel = frame[:, :, 1]
    red_channel = frame[:, :, 2]

    blurred_blue = manual_blur_channel(blue_channel, kernel_size, sigma)
    blurred_green = manual_blur_channel(green_channel, kernel_size, sigma)
    blurred_red = manual_blur_channel(red_channel, kernel_size, sigma)

    output_frame = np.zeros(
        frame.shape,
        dtype=np.uint8
    )

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
        print("[ERROR] Cannot open video:")
        print(input_video_path)
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
        print("[ERROR] Cannot create output video:")
        print(output_video_path)
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

    video_files = []

    for file_name in all_files:

        lower_name = file_name.lower()

        for extension in VIDEO_EXTENSIONS:

            if lower_name.endswith(extension):
                video_files.append(file_name)
                break

    video_files.sort()

    if limit is not None:
        video_files = video_files[:limit]

    total_videos = len(video_files)

    for index in range(total_videos):

        file_name = video_files[index]

        print()
        print(f"========== VIDEO {index + 1}/{total_videos} ==========")
        print(file_name)

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

    parser = argparse.ArgumentParser(
        description="Manual Gaussian blur for surveillance videos."
    )

    parser.add_argument(
        "--input-root",
        type=str,
        default="data/videos_clahe",
        help="Input dataset root (CLAHE-processed videos)"
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="data/videos_gaussian_blur",
        help="Output dataset root"
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=KERNEL_SIZE,
        help="Odd kernel size (e.g. 5)"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA,
        help="Gaussian sigma value"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos per category"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos"
    )

    args = parser.parse_args()

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

    print()
    print("====================================")
    print("GAUSSIAN BLUR PREPROCESSING COMPLETE")
    print("====================================")
