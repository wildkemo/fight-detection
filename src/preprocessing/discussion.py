import os
import sys
import cv2
import glob
import argparse
from pathlib import Path

# Add current directory to path to import siblings
sys.path.append(str(Path(__file__).parent))

from contrast_stretching import manual_contrast_stretch
from clahe import manual_clahe
from resize import manual_letterbox
from guassien_blur import manual_gaussian_blur

def process_videos(input_dir="data/videos", output_dir="data/images", target_fps=3):
    """
    Standalone loop to sample videos at 3 FPS and apply manual preprocessing.
    Saves output as images.
    """
    
    # Supported video extensions
    extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Search for videos in input_dir/*/ (supports category folders)
    video_paths = []
    for ext in extensions:
        video_paths.extend(glob.glob(os.path.join(input_dir, "**", f"*{ext}"), recursive=True))
    
    video_paths.sort()
    
    if not video_paths:
        print(f"No videos found in {input_dir}")
        return

    print(f"Found {len(video_paths)} videos. Starting processing...")

    for video_path in video_paths:
        # Determine category and video name for output path
        rel_path = os.path.relpath(video_path, input_dir)
        category = os.path.dirname(rel_path)
        video_base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        save_dir = os.path.join(output_dir, category, video_base_name)
        os.makedirs(save_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30 # Default if metadata is missing
            
        # Calculate interval to hit target_fps
        frame_interval = max(1, int(round(fps / target_fps)))
        
        frame_idx = 0
        saved_idx = 0
        
        print(f"Processing: {video_path} | Sampling every {frame_interval} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # STEP 1: Manual Contrast Stretching
                processed = manual_contrast_stretch(frame)
                
                # STEP 2: Manual CLAHE
                processed = manual_clahe(processed, tile_grid_x=8, tile_grid_y=8, clip_limit=2.0)
                
                # STEP 3: Manual Letterbox (640x640)
                if processed is not None:
                    processed = manual_letterbox(processed, target_width=640, target_height=640)
                
                # STEP 4: Manual Gaussian Blur (5x5, sigma 1.0)
                if processed is not None:
                    processed = manual_gaussian_blur(processed, kernel_size=5, sigma=1.0)
                
                # Save Result
                if processed is not None:
                    out_name = f"frame_{saved_idx:06d}.jpg"
                    cv2.imwrite(os.path.join(save_dir, out_name), processed)
                    saved_idx += 1
            
            frame_idx += 1
            
        cap.release()
        print(f"Saved {saved_idx} frames to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Video to Sampled Images Preprocessor")
    parser.add_argument("--input", type=str, default="data/videos", help="Root folder of videos")
    parser.add_argument("--output", type=str, default="data/images", help="Folder to save images")
    parser.add_argument("--fps", type=int, default=3, help="Sampling frequency")
    
    args = parser.parse_args()
    
    process_videos(input_dir=args.input, output_dir=args.output, target_fps=args.fps)
    print("\nAll tasks completed.")
