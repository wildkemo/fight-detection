import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_DIR = Path("output/yolo_poses")
OUTPUT_DIR = Path("output/dataset")
SEQUENCE_LENGTH = 36

def main():
    print("Starting ID-based sequence building from YOLO-Pose data...")
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

            # Find all _tracks.json files
            json_files = list(cls_dir.glob("**/*_tracks.json"))
            label = 1 if cls == "Violence" else 0

            for json_file in tqdm(json_files, desc=f" {cls}"):
                with open(json_file, 'r') as f:
                    video_tracks = json.load(f)

                # video_tracks format: { "track_id": [ {"frame_id": int, "keypoints": [...]}, ... ] }
                for track_id, instances in video_tracks.items():
                    # Ensure instances are sorted by frame_id
                    instances = sorted(instances, key=lambda x: x['frame_id'])

                    # Only process if we have enough frames for at least one window
                    if len(instances) >= SEQUENCE_LENGTH:
                        # Extract just the keypoints list
                        all_kpts = [inst['keypoints'] for inst in instances]

                        # Sliding window
                        for i in range(0, len(all_kpts) - SEQUENCE_LENGTH + 1):
                            window = all_kpts[i : i + SEQUENCE_LENGTH]
                            # window is (36, 17, 3)
                            # Flatten: (36, 17, 3) -> (36, 51)
                            flattened_window = np.array(window).reshape(SEQUENCE_LENGTH, 51)
                            X_split.append(flattened_window)
                            y_split.append(label)

        if X_split:
            X_split = np.array(X_split)
            y_split = np.array(y_split)
            # Use a new filename prefix to avoid confusion with old, low-quality data
            np.save(OUTPUT_DIR / f"X_{split}_yolo.npy", X_split)
            np.save(OUTPUT_DIR / f"y_{split}_yolo.npy", y_split)
            print(f" Finished {split}: X shape {X_split.shape}, y shape {y_split.shape}")
        else:
            print(f" No sequences found for {split} split.")

if __name__ == "__main__":
    main()
