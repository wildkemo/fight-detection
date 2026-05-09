import os
import sys
import cv2
from pathlib import Path
from tqdm import tqdm

# Add current directory to path if running directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frame_extractor import extract_frames
from resizer import resize_with_padding
from denoiser import denoise_frame
from clahe import apply_clahe
from normalizer import normalize_frame, denormalize_to_uint8

def process_video(video_path, output_base_dir, category):
    """
    Processes a single video: extracts frames, applies preprocessing, and saves them.
    """
    video_path = Path(video_path)
    video_name = video_path.stem
    output_dir = Path(output_base_dir) / category / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed (basic check)
    if list(output_dir.glob("*.jpg")):
        # Skipping if frames already exist
        return

    for frame_idx, frame in extract_frames(str(video_path), target_fps=5):
        # 1. Resize with padding
        frame = resize_with_padding(frame, target_size=(224, 224))
        
        # 2. Mild Denoising
        frame = denoise_frame(frame)
        
        # 3. CLAHE
        frame = apply_clahe(frame)
        
        # 4. Normalization (and back to uint8 for saving)
        # In a real training pipeline, we might save as float32 tensors,
        # but for visualization and storage efficiency, we save as JPG.
        normalized = normalize_frame(frame)
        final_frame = denormalize_to_uint8(normalized)
        
        # Save frame
        frame_filename = f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(output_dir / frame_filename), final_frame)

def main():
    dataset_dir = Path("dataset")
    output_dir = Path("output")
    
    categories = ["Violence", "NonViolence"]
    
    for category in categories:
        cat_dir = dataset_dir / category
        if not cat_dir.exists():
            print(f"Warning: Category directory {cat_dir} not found. Skipping.")
            continue
            
        videos = list(cat_dir.glob("*.mp4")) + list(cat_dir.glob("*.avi"))
        print(f"Processing {len(videos)} videos in category: {category}")
        
        for video_path in tqdm(videos, desc=f"Category: {category}"):
            try:
                process_video(video_path, output_dir, category)
            except Exception as e:
                print(f"\nError processing {video_path}: {e}")
                continue

if __name__ == "__main__":
    main()
