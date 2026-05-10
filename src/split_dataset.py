import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(output_dir="output", split_dir="data_splits", train_ratio=0.8, val_ratio=0.1):
    """
    Splits video frame folders into train, val, and test sets.
    Physically copies/moves frames into a structure compatible with standard data loaders.
    """
    output_path = Path(output_dir)
    split_path = Path(split_dir)
    
    categories = ["Violence", "NonViolence"]
    
    # Initialize split directories
    splits = ["train", "val", "test"]
    for s in splits:
        for c in categories:
            (split_path / s / c).mkdir(parents=True, exist_ok=True)
            
    for category in categories:
        cat_path = output_path / category
        if not cat_path.exists():
            print(f"Warning: {cat_path} not found. Skipping.")
            continue
            
        video_folders = [f for f in cat_path.iterdir() if f.is_dir()]
        random.shuffle(video_folders)
        
        n_videos = len(video_folders)
        n_train = int(n_videos * train_ratio)
        n_val = int(n_videos * val_ratio)
        
        train_videos = video_folders[:n_train]
        val_videos = video_folders[n_train:n_train + n_val]
        test_videos = video_folders[n_train + n_val:]
        
        split_map = {
            "train": train_videos,
            "val": val_videos,
            "test": test_videos
        }
        
        for split_name, videos in split_map.items():
            print(f"Copying {len(videos)} videos from {category} to {split_name} split...")
            for video_folder in tqdm(videos, desc=f"{split_name}/{category}"):
                # To maintain video-level split while being compatible with Keras ImageFolder:
                # We can either flatten or keep the video folder if the loader supports it.
                # Keras image_dataset_from_directory searches subdirectories recursively.
                
                target_category_dir = split_path / split_name / category / video_folder.name
                # Use symlinks for efficiency if the OS supports it, otherwise copy.
                # Since we are in a Linux environment, symlinks are great.
                # However, the user asked for "Physical Directory Split", so I'll copy.
                shutil.copytree(video_folder, target_category_dir, dirs_exist_ok=True)

if __name__ == "__main__":
    split_dataset()
