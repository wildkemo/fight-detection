import cv2
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from src.preprocessors.denoiser import denoise_frame
from src.preprocessors.clahe import apply_clahe
from src.preprocessors.normalizer import normalize_frame, denormalize_to_uint8

# Configuration
DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("output/frames")
TARGET_FPS = 30
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
CLASSES = ["Violence", "NonViolence"]

def get_video_files(directory):
    """Returns a list of all mp4 files in the directory."""
    return list(directory.glob("*.mp4"))

def split_videos(video_list, splits):
    """Splits a list of videos into train, val, and test sets."""
    random.seed(42)  # For reproducibility
    random.shuffle(video_list)
    
    total = len(video_list)
    train_end = int(total * splits["train"])
    val_end = train_end + int(total * splits["val"])
    
    return {
        "train": video_list[:train_end],
        "val": video_list[train_end:val_end],
        "test": video_list[val_end:]
    }

def extract_frames(video_path, output_path, target_fps=30, desc="Extracting"):
    """Extracts frames from a video at a target FPS."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_fps == 0:
        print(f"Warning: FPS is 0 for {video_path}, skipping.")
        cap.release()
        return

    hop = max(1, round(video_fps / target_fps))
    output_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    frame_id = 0
    
    with tqdm(total=total_frames, desc=desc, leave=False, unit="fr") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % hop == 0:
                frame_filename = output_path / f"frame_{frame_id:04d}.jpg"
                
                # Step 1: Denoise
                processed_tensor = denoise_frame(frame)
                
                # Step 2: Enhance Contrast (CLAHE)
                processed_tensor = apply_clahe(processed_tensor)
                
                # Step 3: Normalize and Denormalize (for uint8 jpg saving)
                processed_tensor = normalize_frame(processed_tensor)
                processed_tensor = denormalize_to_uint8(processed_tensor)
                
                # Final conversion to NumPy for saving
                processed_numpy = processed_tensor.numpy()
                cv2.imwrite(str(frame_filename), processed_numpy)
                frame_id += 1
            
            count += 1
            pbar.update(1)
    
    cap.release()

def print_summary_table(summary):
    """Prints a formatted summary table of the processed dataset."""
    print("\n" + "="*50)
    print(f"{'CLASS':<15} | {'TRAIN':<8} | {'VAL':<8} | {'TEST':<8}")
    print("-" * 50)
    for cls, counts in summary.items():
        print(f"{cls:<15} | {counts.get('train', 0):<8} | {counts.get('val', 0):<8} | {counts.get('test', 0):<8}")
    print("="*50 + "\n")

def main():
    print("\n" + "!"*60)
    print("!!! FIGHT DETECTION SYSTEM - DATA PREPROCESSING PIPELINE !!!")
    print("!"*60 + "\n")
    
    # Ensure output directory is clean
    if OUTPUT_DIR.exists():
        print(f"[*] Cleaning existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    dataset_summary = {}

    for cls in CLASSES:
        cls_dir = DATASET_DIR / cls
        if not cls_dir.exists():
            print(f"[!] Warning: Directory {cls_dir} does not exist. Skipping.")
            continue
            
        videos = get_video_files(cls_dir)
        print(f"[*] Found {len(videos)} videos in class: {cls}")
        
        splits = split_videos(videos, SPLITS)
        dataset_summary[cls] = {k: len(v) for k, v in splits.items()}
        
        for split_name, split_videos_list in splits.items():
            desc = f"[*] Processing {cls} ({split_name})"
            for video_path in tqdm(split_videos_list, desc=desc, unit="vid"):
                video_name = video_path.stem
                video_output_dir = OUTPUT_DIR / split_name / cls / video_name
                extract_frames(video_path, video_output_dir, TARGET_FPS, desc=f"    > {video_name}")

    print("[+] Preprocessing complete!")
    print_summary_table(dataset_summary)

if __name__ == "__main__":
    main()
