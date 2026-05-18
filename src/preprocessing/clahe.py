"""
Manual CLAHE pipeline for videos before detection/tracking.

Goal:
- Improve local contrast using Contrast Limited Adaptive Histogram Equalization
- Build histograms, clip them, and map pixels without cv2.createCLAHE
- Process videos frame-by-frame in the same style as resize.py

Pipeline:
Video -> Read Frame -> Luminance -> Tile Histograms -> Clip -> LUT -> Interpolate -> Save
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

def manual_clip_histogram(histogram, clip_limit, tile_pixel_count):
    """
    Clip histogram bins and redistribute the clipped excess evenly using numpy.
    """
    clip_threshold = (clip_limit * float(tile_pixel_count)) / 256.0
    
    clipped = np.minimum(histogram, clip_threshold)
    excess = np.sum(histogram - clipped)
    redistribution = excess / 256.0
    
    return clipped + redistribution


def manual_build_equalization_lut(histogram):
    """
    Build a lookup table from a (possibly clipped) histogram using CDF mapping.
    """
    cumulative = np.cumsum(histogram)
    total_pixels = cumulative[-1]

    if total_pixels <= 0.0:
        return np.arange(256, dtype=np.uint8)

    # Find minimum CDF value where histogram > 0
    nonzero_indices = np.where(histogram > 0)[0]
    if len(nonzero_indices) == 0:
        return np.arange(256, dtype=np.uint8)
        
    min_idx = nonzero_indices[0]
    minimum_cdf = cumulative[min_idx] - histogram[min_idx]
    
    denominator = total_pixels - minimum_cdf
    if denominator <= 0:
        return np.arange(256, dtype=np.uint8)

    lut = ((cumulative - minimum_cdf) * 255.0 / denominator + 0.5)
    return np.clip(lut, MIN_OUTPUT, MAX_OUTPUT).astype(np.uint8)


def manual_apply_clahe_channel(channel, tile_grid_x, tile_grid_y, clip_limit):
    """
    Apply CLAHE to one channel using bilinear interpolation between tile LUTs.
    Fully vectorized with numpy.
    """
    h, w = channel.shape
    tile_h = h // tile_grid_y
    tile_w = w // tile_grid_x

    # 1. Build LUTs for each tile
    luts = np.empty((tile_grid_y, tile_grid_x, 256), dtype=np.uint8)
    for ty in range(tile_grid_y):
        y_start = ty * tile_h
        y_end = h if ty == tile_grid_y - 1 else (ty + 1) * tile_h
        for tx in range(tile_grid_x):
            x_start = tx * tile_w
            x_end = w if tx == tile_grid_x - 1 else (tx + 1) * tile_w
            
            region = channel[y_start:y_end, x_start:x_end]
            hist = np.bincount(region.ravel(), minlength=256).astype(np.float32)
            
            clipped_hist = manual_clip_histogram(hist, clip_limit, region.size)
            luts[ty, tx] = manual_build_equalization_lut(clipped_hist)

    # 2. Vectorized Bilinear Interpolation
    # Create pixel coordinate grids
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    
    # Calculate fractional tile positions
    # Center of tile (0,0) is at (tile_h/2, tile_w/2)
    fx = (x_coords + 0.5) / tile_w - 0.5
    fy = (y_coords + 0.5) / tile_h - 0.5
    
    # Floor to get the left/top tile indices
    tx_l = np.floor(fx).astype(np.int32)
    ty_t = np.floor(fy).astype(np.int32)
    
    # Clamp to grid bounds
    tx_l = np.clip(tx_l, 0, tile_grid_x - 1)
    ty_t = np.clip(ty_t, 0, tile_grid_y - 1)
    tx_r = np.clip(tx_l + 1, 0, tile_grid_x - 1)
    ty_b = np.clip(ty_t + 1, 0, tile_grid_y - 1)
    
    # Calculate weights
    wx_r = np.clip(fx - tx_l, 0, 1)
    wy_b = np.clip(fy - ty_t, 0, 1)
    wx_l = 1.0 - wx_r
    wy_t = 1.0 - wy_b
    
    # Use advanced indexing to pull values from LUTs
    # luts is [ty, tx, intensity]
    # We need luts[ty_t, tx_l, channel[y, x]], etc.
    
    # Map intensities through the 4 neighboring LUTs
    vals_tl = luts[ty_t, tx_l, channel]
    vals_tr = luts[ty_t, tx_r, channel]
    vals_bl = luts[ty_b, tx_l, channel]
    vals_br = luts[ty_b, tx_r, channel]
    
    # Blend
    top_blend = vals_tl * wx_l + vals_tr * wx_r
    bottom_blend = vals_bl * wx_l + vals_br * wx_r
    final = top_blend * wy_t + bottom_blend * wy_b
    
    return np.clip(np.round(final), MIN_OUTPUT, MAX_OUTPUT).astype(np.uint8)


def manual_clahe(frame, tile_grid_x, tile_grid_y, clip_limit):
    """
    Apply manual CLAHE on luminance and merge back into BGR.
    """
    if frame is None:
        return None

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        print("[ERROR] Expected a 3-channel BGR frame.")
        return frame

    # Vectorized BGR to Luminance
    # luma = 0.299R + 0.587G + 0.114B
    luma = (BLUE_WEIGHT * frame[:, :, 0] + 
            GREEN_WEIGHT * frame[:, :, 1] + 
            RED_WEIGHT * frame[:, :, 2] + 0.5).astype(np.uint8)

    enhanced_luma = manual_apply_clahe_channel(luma, tile_grid_x, tile_grid_y, clip_limit)

    # Re-apply color by scaling
    # Handle zero luma to avoid division by zero
    mask = (luma > 0)
    scale = np.ones_like(luma, dtype=np.float32)
    scale[mask] = enhanced_luma[mask].astype(np.float32) / luma[mask]
    
    # If luma is 0, we can't scale, so we set channels directly to enhanced_luma
    output = np.empty_like(frame)
    for c in range(3):
        res = frame[:, :, c].astype(np.float32) * scale
        res[~mask] = enhanced_luma[~mask]
        output[:, :, c] = np.clip(np.round(res), MIN_OUTPUT, MAX_OUTPUT).astype(np.uint8)

    return output


# =========================================================
# VIDEO PROCESSING
# =========================================================

def clahe_video(input_video_path, output_video_path, tile_grid_x, tile_grid_y, clip_limit, overwrite=False):
    if os.path.exists(output_video_path) and not overwrite:
        print(f"[SKIPPED] {output_video_path}")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = os.path.dirname(output_video_path)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

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

        output_frame = manual_clahe(frame, tile_grid_x, tile_grid_y, clip_limit)
        writer.write(output_frame)
        current_frame += 1
        print(f"\rProcessing: {os.path.basename(input_video_path)} [{current_frame}/{frame_count}]", end="")

    print()
    cap.release()
    writer.release()
    print(f"[DONE] {output_video_path}")


# =========================================================
# DIRECTORY PROCESSING
# =========================================================

def process_directory(input_directory, output_directory, tile_grid_x, tile_grid_y, clip_limit, limit=None, overwrite=False):
    os.makedirs(output_directory, exist_ok=True)
    video_files = [f for f in os.listdir(input_directory) if any(f.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)]
    video_files.sort()

    if limit is not None:
        video_files = video_files[:limit]

    total_videos = len(video_files)
    for index, file_name in enumerate(video_files):
        print(f"\n========== VIDEO {index + 1}/{total_videos} ==========\n{file_name}")
        input_path = os.path.join(input_directory, file_name)
        output_path = os.path.join(output_directory, file_name)
        clahe_video(input_path, output_path, tile_grid_x, tile_grid_y, clip_limit, overwrite)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual CLAHE for surveillance videos.")
    parser.add_argument("--input-root", type=str, default="data/videos_contrast_stretched")
    parser.add_argument("--output-root", type=str, default="data/videos_clahe")
    parser.add_argument("--clip-limit", type=float, default=CLIP_LIMIT)
    parser.add_argument("--tiles-x", type=int, default=TILE_GRID_X)
    parser.add_argument("--tiles-y", type=int, default=TILE_GRID_Y)
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

        process_directory(input_dir, output_dir, args.tiles_x, args.tiles_y, args.clip_limit, args.limit, args.overwrite)

    print("\n====================================\nCLAHE PREPROCESSING COMPLETE\n====================================")