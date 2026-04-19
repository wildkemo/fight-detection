import os
import sys
from extract_frames import extract_frames
from resize import resize_frame
from denoise import denoise_frame
from dynamic_range import apply_dynamic_range_adjustment
from contrast import apply_contrast_enhancement
from edge_enhancement import apply_edge_enhancement
from color_space import optimize_color_space
from sanity_check import sanity_check
from storage import save_frame_organized

# Configuration
VIDEO_PATH = "dataset/Violence/V_31.mp4"
OUTPUT_ROOT = "output"
BURST_SIZE = 16  # Group frames into sets of 16

def run_pipeline():
    """
    Orchestrates the 9-step preprocessing pipeline with enhanced logging.
    """
    print("\n" + "═"*60)
    print(" 🎬 CCTV VIOLENCE DETECTION PREPROCESSING ".center(60))
    print("═"*60)

    if not os.path.exists(VIDEO_PATH):
        print(f" ❌ Error: Video file '{VIDEO_PATH}' not found.")
        print("═"*60 + "\n")
        return

    print(f" 📂 Input Video : {VIDEO_PATH}")
    print(f" 📁 Output Root : {OUTPUT_ROOT}")
    print("─"*60)
    
    # Step 1: Extract Frames
    print(f" 🎞️  Step 1: Extracting frames...", end="", flush=True)
    frames = extract_frames(VIDEO_PATH)
    print(f" Done. ({len(frames)} frames)")
    
    print(f" 🪄  Steps 2-7: Applying image enhancement...")
    
    saved_count = 0
    total_frames = len(frames)
    
    for i, frame in enumerate(frames):
        # Step 2: Resize and Standardize (Skipped)
        # frame = resize_frame(frame)

        # Step 3: Denoising
        frame = denoise_frame(frame)
        
        # Step 4: Dynamic Range Adjustment (Skipped)
        # frame = apply_dynamic_range_adjustment(frame)
        
        # Step 5: Contrast Enhancement
        frame = apply_contrast_enhancement(frame)
        
        # Step 6: Edge Enhancement
        frame = apply_edge_enhancement(frame)
        
        # Step 7: Color Space Optimization
        frame = optimize_color_space(frame, to_grayscale=True)

        # Step 8: Sanity Check
        if sanity_check(frame):
            # Step 9: Organized Storage
            save_frame_organized(frame, VIDEO_PATH, OUTPUT_ROOT, saved_count, BURST_SIZE)
            saved_count += 1
        
        # Simple Progress Indicator
        progress = (i + 1) / total_frames
        bar_length = 30
        filled = int(progress * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        sys.stdout.write(f"\r    [{bar}] {i+1}/{total_frames} frames processed")
        sys.stdout.flush()
            
    print(f"\n" + "─"*60)
    print(" ✨ PREPROCESSING COMPLETE ".center(60))
    print(f" 📊 Total Frames Saved : {saved_count}")
    print(f" 📦 Bursts Created     : {(saved_count + BURST_SIZE - 1) // BURST_SIZE}")
    print("═"*60 + "\n")

if __name__ == "__main__":
    run_pipeline()
