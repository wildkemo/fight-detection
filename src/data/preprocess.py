import os
import sys
import glob
from extract_frames import extract_frames
from resize import resize
from denoise import denoise_guided, median_blur
from dynamic_range import dynamic_range
from contrast import contrast
from sharpen import sharpen
from color_space import color_space
from sanity_check import sanity_check
from storage import storage

# Configuration
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DATASET_DIR = BASE_DIR / "dataset" 
OUTPUT_ROOT = PROJECT_ROOT / "output"
BURST_SIZE = 16  # Group frames into sets of 16
USE_SINGLE_VIDEO = False  # Set to True to process only one video
SINGLE_VIDEO_PATH = str(DATASET_DIR / "Violence" / "V_515.mp4")  # Path used when USE_SINGLE_VIDEO is True

def process_video(video_path):
    """
    Runs the enhancement pipeline on a single video.
    Returns the number of frames saved.
    """
    print(f"\n 📂 Processing Video : {os.path.basename(video_path)}")
    print("─"*60)
    
    # Step 1: Extract Frames
    print(f" 🎞️  Step 1: Extracting frames...", end="", flush=True)
    frames = extract_frames(video_path)
    print(f" Done. ({len(frames)} frames)")
    
    if not frames:
        print(" ⚠️  Warning: No frames extracted.")
        return 0

    print(f" 🪄  Steps 2-7: Applying image enhancement...")
    
    saved_count = 0
    total_frames = len(frames)
    
    for i, frame in enumerate(frames):
        # --- PHASE 1: CONVERSION TO GRAYSCALE ---
        # Convert to grayscale immediately for maximum processing speed in all steps
        frame = color_space(frame)

        # Step 2: Resize and Standardize (Skipped)
        # frame = resize(frame)

        # Step 3: Denoising (Combined Median + Guided)
        frame = median_blur(frame, kernel_size=3)
        frame = denoise_guided(frame, r=1, eps=1e-2)

        # Step 4: contrast stretching
        frame = dynamic_range(frame)

        # Step 5: Contrast Enhancement
        frame = contrast(frame)
        
        # Step 6: Sharpening 
        frame = sharpen(frame)

        

        # Step 8: Sanity Check
        if sanity_check(frame):
            # Step 9: Organized Storage
            storage(frame, video_path, OUTPUT_ROOT, saved_count, BURST_SIZE)
            saved_count += 1
        
        # Simple Progress Indicator
        progress = (i + 1) / total_frames
        bar_length = 30
        filled = int(progress * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        sys.stdout.write(f"\r    [{bar}] {i+1}/{total_frames} frames processed")
        sys.stdout.flush()
        
    print(f"\n    ✅ Saved {saved_count} frames.")
    return saved_count

def run_pipeline():
    """
    Orchestrates the preprocessing pipeline across an entire directory.
    """
    print("\n" + "═" * 60)
    print(" 🎬 CCTV VIOLENCE DETECTION BATCH PREPROCESSING ".center(60))
    print("═" * 60)

    if not DATASET_DIR.exists():
        print(f" ❌ Error: Dataset directory '{DATASET_DIR}' not found.")
        print("═" * 60 + "\n")
        return

    # Automatically create OUTPUT_ROOT if it doesn't exist
    if not OUTPUT_ROOT.exists():
        print(f" 📂 Creating output directory: {OUTPUT_ROOT}")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if USE_SINGLE_VIDEO:
        video_files = [SINGLE_VIDEO_PATH]
        print(f" ℹ️  Single video mode: {SINGLE_VIDEO_PATH}")
    else:
        video_files = glob.glob(str(DATASET_DIR / "*" / "*.mp4"))
        if not video_files:
            print(f" ⚠️  No .mp4 files found in '{DATASET_DIR}'.")
            print("═" * 60 + "\n")
            return

        # Sort video files to process "Violence" first, then "NonViolence"
        video_files.sort(key=lambda x: (0 if os.path.basename(os.path.dirname(x)) == "Violence" else 1, x))

    print(f" 📁 Dataset Path: {DATASET_DIR}")
    print(f" 📁 Output Root : {OUTPUT_ROOT}")
    print(f" 📹 Videos Found: {len(video_files)}")

    total_saved_frames = 0

    for video_path in video_files:
        saved_frames = process_video(video_path)
        total_saved_frames += saved_frames

    print("\n" + "═" * 60)
    print(" ✨ BATCH PREPROCESSING COMPLETE ".center(60))
    print(f" 📊 Total Videos Processed : {len(video_files)}")
    print(f" 📊 Total Frames Saved     : {total_saved_frames}")
    print("═" * 60 + "\n")
    
if __name__ == "__main__":
    run_pipeline()
