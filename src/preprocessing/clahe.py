"""
Manual CLAHE pipeline for videos before detection/tracking.

Goal:
- Improve local contrast using Contrast Limited Adaptive Histogram Equalization
- Build histograms, clip them, and map pixels without cv2.createCLAHE
- Process videos frame-by-frame in the same style as resize.py

Pipeline:
Video -> Read Frame -> Luminance -> Tile Histograms -> Clip -> LUT -> Interpolate -> Save

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

CLIP_LIMIT = 2.0
TILE_GRID_X = 8
TILE_GRID_Y = 8

MIN_OUTPUT = 0
MAX_OUTPUT = 255

# ITU-R BT.601 luminance weights for manual BGR -> Y conversion
BLUE_WEIGHT = 0.114
GREEN_WEIGHT = 0.587
RED_WEIGHT = 0.299


# =========================================================
# MANUAL HISTOGRAM + CLAHE HELPERS
# =========================================================

def manual_build_histogram_for_region(channel, row_start, row_end, col_start, col_end):
    """
    Count pixel intensities inside a rectangular tile region.
    """

    histogram = [0] * 256

    for row_index in range(row_start, row_end):

        for col_index in range(col_start, col_end):

            intensity = int(channel[row_index, col_index])

            histogram[intensity] += 1

    return histogram


def manual_clip_histogram(histogram, clip_limit, tile_pixel_count):
    """
    Clip histogram bins and redistribute the clipped excess evenly.
    """

    clipped_histogram = []

    for bin_index in range(256):
        clipped_histogram.append(float(histogram[bin_index]))

    clip_threshold = (clip_limit * float(tile_pixel_count)) / 256.0

    excess = 0.0

    for bin_index in range(256):

        if clipped_histogram[bin_index] > clip_threshold:
            excess += clipped_histogram[bin_index] - clip_threshold
            clipped_histogram[bin_index] = clip_threshold

    redistribution = excess / 256.0

    for bin_index in range(256):
        clipped_histogram[bin_index] += redistribution

    return clipped_histogram


def manual_build_equalization_lut(histogram):
    """
    Build a lookup table from a (possibly clipped) histogram using CDF mapping.
    """

    lookup_table = [0] * 256

    cumulative = [0.0] * 256
    running_sum = 0.0
    total_pixels = 0.0

    for bin_index in range(256):
        running_sum += histogram[bin_index]
        cumulative[bin_index] = running_sum
        total_pixels += histogram[bin_index]

    if total_pixels <= 0.0:
        for bin_index in range(256):
            lookup_table[bin_index] = bin_index
        return lookup_table

    minimum_cdf = 0.0
    found_minimum = False

    for bin_index in range(256):

        if histogram[bin_index] > 0.0 and not found_minimum:
            minimum_cdf = cumulative[bin_index] - histogram[bin_index]
            found_minimum = True
            break

    denominator = total_pixels - minimum_cdf

    for bin_index in range(256):

        if denominator > 0.0:
            mapped_value = int(
                ((cumulative[bin_index] - minimum_cdf) * 255.0) / denominator + 0.5
            )
        else:
            mapped_value = bin_index

        if mapped_value < MIN_OUTPUT:
            mapped_value = MIN_OUTPUT
        elif mapped_value > MAX_OUTPUT:
            mapped_value = MAX_OUTPUT

        lookup_table[bin_index] = mapped_value

    return lookup_table


def manual_build_tile_lookup_tables(channel, tile_grid_x, tile_grid_y, clip_limit):
    """
    Compute one equalization LUT for every tile in the image grid.
    """

    image_height = channel.shape[0]
    image_width = channel.shape[1]

    tile_width = image_width // tile_grid_x
    tile_height = image_height // tile_grid_y

    if tile_width < 1:
        tile_width = 1

    if tile_height < 1:
        tile_height = 1

    tile_lookup_tables = []

    for tile_row in range(tile_grid_y):

        row_tables = []

        for tile_col in range(tile_grid_x):

            row_start = tile_row * tile_height
            col_start = tile_col * tile_width

            if tile_row == tile_grid_y - 1:
                row_end = image_height
            else:
                row_end = row_start + tile_height

            if tile_col == tile_grid_x - 1:
                col_end = image_width
            else:
                col_end = col_start + tile_width

            tile_histogram = manual_build_histogram_for_region(
                channel,
                row_start,
                row_end,
                col_start,
                col_end
            )

            tile_pixel_count = (row_end - row_start) * (col_end - col_start)

            clipped_histogram = manual_clip_histogram(
                tile_histogram,
                clip_limit,
                tile_pixel_count
            )

            tile_lut = manual_build_equalization_lut(clipped_histogram)
            row_tables.append(tile_lut)

        tile_lookup_tables.append(row_tables)

    return tile_lookup_tables, tile_width, tile_height


def manual_lookup_value(lookup_table, intensity):
    """
    Read a mapped intensity from a LUT.
    """

    if intensity < 0:
        intensity = 0
    elif intensity > 255:
        intensity = 255

    return lookup_table[intensity]


def manual_apply_clahe_channel(channel, tile_grid_x, tile_grid_y, clip_limit):
    """
    Apply CLAHE to one channel using bilinear interpolation between tile LUTs.
    """

    image_height = channel.shape[0]
    image_width = channel.shape[1]

    tile_lookup_tables, tile_width, tile_height = manual_build_tile_lookup_tables(
        channel,
        tile_grid_x,
        tile_grid_y,
        clip_limit
    )

    output_channel = np.zeros(
        (image_height, image_width),
        dtype=np.uint8
    )

    for row_index in range(image_height):

        for col_index in range(image_width):

            tile_x = (float(col_index) + 0.5) / float(tile_width) - 0.5
            tile_y = (float(row_index) + 0.5) / float(tile_height) - 0.5

            left_tile = int(tile_x)
            top_tile = int(tile_y)
            right_tile = left_tile + 1
            bottom_tile = top_tile + 1

            if left_tile < 0:
                left_tile = 0
            if top_tile < 0:
                top_tile = 0
            if right_tile >= tile_grid_x:
                right_tile = tile_grid_x - 1
            if bottom_tile >= tile_grid_y:
                bottom_tile = tile_grid_y - 1
            if left_tile >= tile_grid_x:
                left_tile = tile_grid_x - 1
            if top_tile >= tile_grid_y:
                top_tile = tile_grid_y - 1

            weight_right = tile_x - float(left_tile)
            weight_bottom = tile_y - float(top_tile)

            if weight_right < 0.0:
                weight_right = 0.0
            elif weight_right > 1.0:
                weight_right = 1.0

            if weight_bottom < 0.0:
                weight_bottom = 0.0
            elif weight_bottom > 1.0:
                weight_bottom = 1.0

            weight_left = 1.0 - weight_right
            weight_top = 1.0 - weight_bottom

            intensity = int(channel[row_index, col_index])

            top_left = manual_lookup_value(
                tile_lookup_tables[top_tile][left_tile],
                intensity
            )
            top_right = manual_lookup_value(
                tile_lookup_tables[top_tile][right_tile],
                intensity
            )
            bottom_left = manual_lookup_value(
                tile_lookup_tables[bottom_tile][left_tile],
                intensity
            )
            bottom_right = manual_lookup_value(
                tile_lookup_tables[bottom_tile][right_tile],
                intensity
            )

            top_blend = (
                float(top_left) * weight_left +
                float(top_right) * weight_right
            )
            bottom_blend = (
                float(bottom_left) * weight_left +
                float(bottom_right) * weight_right
            )

            final_value = int(top_blend * weight_top + bottom_blend * weight_bottom + 0.5)

            if final_value < MIN_OUTPUT:
                final_value = MIN_OUTPUT
            elif final_value > MAX_OUTPUT:
                final_value = MAX_OUTPUT

            output_channel[row_index, col_index] = final_value

    return output_channel


def manual_bgr_to_luminance(frame):
    """
    Convert a BGR frame to a single luminance channel (manual weighted sum).
    """

    image_height = frame.shape[0]
    image_width = frame.shape[1]

    luminance = np.zeros(
        (image_height, image_width),
        dtype=np.uint8
    )

    for row_index in range(image_height):

        for col_index in range(image_width):

            blue_value = float(frame[row_index, col_index, 0])
            green_value = float(frame[row_index, col_index, 1])
            red_value = float(frame[row_index, col_index, 2])

            luminance_value = int(
                BLUE_WEIGHT * blue_value +
                GREEN_WEIGHT * green_value +
                RED_WEIGHT * red_value +
                0.5
            )

            if luminance_value < MIN_OUTPUT:
                luminance_value = MIN_OUTPUT
            elif luminance_value > MAX_OUTPUT:
                luminance_value = MAX_OUTPUT

            luminance[row_index, col_index] = luminance_value

    return luminance


def manual_apply_luminance_clahe_to_bgr(frame, enhanced_luminance, original_luminance):
    """
    Keep color by scaling each BGR channel according to luminance change.
    """

    image_height = frame.shape[0]
    image_width = frame.shape[1]

    output_frame = np.zeros(
        frame.shape,
        dtype=np.uint8
    )

    for row_index in range(image_height):

        for col_index in range(image_width):

            old_luma = int(original_luminance[row_index, col_index])
            new_luma = int(enhanced_luminance[row_index, col_index])

            blue_value = int(frame[row_index, col_index, 0])
            green_value = int(frame[row_index, col_index, 1])
            red_value = int(frame[row_index, col_index, 2])

            if old_luma > 0:
                scale = float(new_luma) / float(old_luma)

                blue_value = int(blue_value * scale + 0.5)
                green_value = int(green_value * scale + 0.5)
                red_value = int(red_value * scale + 0.5)
            else:
                blue_value = new_luma
                green_value = new_luma
                red_value = new_luma

            if blue_value < MIN_OUTPUT:
                blue_value = MIN_OUTPUT
            elif blue_value > MAX_OUTPUT:
                blue_value = MAX_OUTPUT

            if green_value < MIN_OUTPUT:
                green_value = MIN_OUTPUT
            elif green_value > MAX_OUTPUT:
                green_value = MAX_OUTPUT

            if red_value < MIN_OUTPUT:
                red_value = MIN_OUTPUT
            elif red_value > MAX_OUTPUT:
                red_value = MAX_OUTPUT

            output_frame[row_index, col_index, 0] = blue_value
            output_frame[row_index, col_index, 1] = green_value
            output_frame[row_index, col_index, 2] = red_value

    return output_frame


def manual_clahe(frame, tile_grid_x, tile_grid_y, clip_limit):
    """
    Apply manual CLAHE on luminance and merge back into BGR.
    """

    if frame is None:
        return None

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        print("[ERROR] Expected a 3-channel BGR frame.")
        return frame

    original_luminance = manual_bgr_to_luminance(frame)

    enhanced_luminance = manual_apply_clahe_channel(
        original_luminance,
        tile_grid_x,
        tile_grid_y,
        clip_limit
    )

    output_frame = manual_apply_luminance_clahe_to_bgr(
        frame,
        enhanced_luminance,
        original_luminance
    )

    return output_frame


# =========================================================
# VIDEO PROCESSING
# =========================================================

def clahe_video(
    input_video_path,
    output_video_path,
    tile_grid_x,
    tile_grid_y,
    clip_limit,
    overwrite=False
):
    """
    Read a video frame-by-frame, apply manual CLAHE, and save output.
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

        output_frame = manual_clahe(
            frame,
            tile_grid_x,
            tile_grid_y,
            clip_limit
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
    tile_grid_x,
    tile_grid_y,
    clip_limit,
    limit=None,
    overwrite=False
):
    """
    Apply CLAHE to all videos inside a directory.
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

        clahe_video(
            input_path,
            output_path,
            tile_grid_x,
            tile_grid_y,
            clip_limit,
            overwrite
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Manual CLAHE for surveillance videos."
    )

    parser.add_argument(
        "--input-root",
        type=str,
        default="data/videos_contrast_stretched",
        help="Input dataset root (contrast-stretched videos)"
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="data/videos_clahe",
        help="Output dataset root"
    )

    parser.add_argument(
        "--clip-limit",
        type=float,
        default=CLIP_LIMIT,
        help="Contrast clipping limit (typical: 2.0)"
    )

    parser.add_argument(
        "--tiles-x",
        type=int,
        default=TILE_GRID_X,
        help="Number of tiles along image width"
    )

    parser.add_argument(
        "--tiles-y",
        type=int,
        default=TILE_GRID_Y,
        help="Number of tiles along image height"
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
            args.tiles_x,
            args.tiles_y,
            args.clip_limit,
            args.limit,
            args.overwrite
        )

    print()
    print("====================================")
    print("CLAHE PREPROCESSING COMPLETE")
    print("====================================")
