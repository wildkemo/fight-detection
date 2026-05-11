import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import distance as dist

# Configuration
INPUT_DIR = Path("output/poses")
OUTPUT_DIR = Path("output/dataset")
SEQUENCE_LENGTH = 16
DISTANCE_THRESHOLD = 100  # Max distance to link centers between frames

def get_center(pose):
    """Calculates the center of a person based on their keypoints."""
    # MoveNet keypoints are [y, x, score]
    # Filter points with reasonable confidence
    valid_points = pose[pose[:, 2] > 0.1]
    if len(valid_points) == 0:
        return None
    return np.mean(valid_points[:, :2], axis=0)

def main():
    print("Starting sequence building and tracking...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    splits = ["train", "val", "test"]
    classes = ["Violence", "NonViolence"]
    
    for split in splits:
        print(f"Processing {split} split...")
        X_split = []
        y_split = []
        
        for cls in classes:
            cls_dir = INPUT_DIR / split / cls
            if not cls_dir.exists(): continue
            
            video_folders = [f for f in cls_dir.iterdir() if f.is_dir()]
            label = 1 if cls == "Violence" else 0
            
            for video_folder in tqdm(video_folders, desc=f" {cls}"):
                # Group files by frame
                pose_files = sorted(list(video_folder.glob("*.npy")))
                frames_data = {}
                for f in pose_files:
                    # Filename format: frame_XXXX_person_NN_pose.npy
                    frame_id = f.name.split("_")[1]
                    if frame_id not in frames_data:
                        frames_data[frame_id] = []
                    frames_data[frame_id].append(np.load(f))
                
                # Tracking Logic
                active_tracks = [] # List of lists: [ [pose_f1, pose_f2, ...], [pose_f1, ...]]
                sorted_frame_ids = sorted(frames_data.keys())
                
                # Previous centers for matching
                prev_centers = [] # List of centers matching active_tracks indices
                
                for f_id in sorted_frame_ids:
                    current_poses = frames_data[f_id]
                    current_centers = [get_center(p) for p in current_poses]
                    current_centers = [c for c in current_centers if c is not None]
                    
                    if not active_tracks:
                        # Initial frame: every person starts a track
                        for p, c in zip(current_poses, current_centers):
                            active_tracks.append([p])
                            prev_centers.append(c)
                    else:
                        # Match current centers to previous centers
                        if current_centers:
                            D = dist.cdist(np.array(prev_centers), np.array(current_centers))
                            rows = D.min(axis=1).argsort()
                            cols = D.argmin(axis=1)[rows]
                            
                            used_rows = set()
                            used_cols = set()
                            
                            for (row, col) in zip(rows, cols):
                                if row in used_rows or col in used_cols:
                                    continue
                                if D[row, col] > DISTANCE_THRESHOLD:
                                    continue
                                
                                # Match found! Append to existing track
                                active_tracks[row].append(current_poses[col])
                                prev_centers[row] = current_centers[col]
                                used_rows.add(row)
                                used_cols.add(col)
                            
                            # Remaining current centers start new tracks
                            for i, c in enumerate(current_centers):
                                if i not in used_cols:
                                    active_tracks.append([current_poses[i]])
                                    prev_centers.append(c)
                    
                    # Optional: Could add logic to "kill" tracks that aren't matched for X frames.
                    # For simplicity, we keep them until the video ends.

                # Windowing and Flattening
                for track in active_tracks:
                    if len(track) >= SEQUENCE_LENGTH:
                        # Create sliding windows
                        for i in range(0, len(track) - SEQUENCE_LENGTH + 1):
                            window = track[i : i + SEQUENCE_LENGTH]
                            # Flatten: (16, 17, 3) -> (16, 51)
                            flattened_window = np.array(window).reshape(SEQUENCE_LENGTH, 51)
                            X_split.append(flattened_window)
                            y_split.append(label)

        if X_split:
            X_split = np.array(X_split)
            y_split = np.array(y_split)
            np.save(OUTPUT_DIR / f"X_{split}.npy", X_split)
            np.save(OUTPUT_DIR / f"y_{split}.npy", y_split)
            print(f" Finished {split}: X shape {X_split.shape}, y shape {y_split.shape}")
        else:
            print(f" No sequences found for {split} split.")

if __name__ == "__main__":
    main()
