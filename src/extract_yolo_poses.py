import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
INPUT_DIR = Path("output/frames")
POSE_OUTPUT_DIR = Path("output/yolo_poses")
VIS_OUTPUT_DIR = Path("output/visualizations")
MODEL_PATH = "yolov8s-pose.pt"
FPS = 12

def main():
    print("Loading YOLOv8n-Pose model...")
    model = YOLO(MODEL_PATH)
    
    POSE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    video_folders = list(INPUT_DIR.glob("**/*/"))
    video_folders = [f for f in video_folders if any(f.glob("*.jpg"))]
    
    print(f"Found {len(video_folders)} videos to process.")

    for video_folder in tqdm(video_folders, desc="Processing videos"):
        rel_path = video_folder.relative_to(INPUT_DIR)
        
        # Prepare output directories
        pose_dir = POSE_OUTPUT_DIR / rel_path
        pose_dir.mkdir(parents=True, exist_ok=True)
        
        vis_dir = VIS_OUTPUT_DIR / rel_path.parent
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        frames = sorted(list(video_folder.glob("*.jpg")))
        if not frames: continue
        
        # Setup VideoWriter
        first_frame = cv2.imread(str(frames[0]))
        h, w = first_frame.shape[:2]
        vis_path = vis_dir / f"{video_folder.name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(vis_path), fourcc, FPS, (w, h))
        
        # Dictionary to store all tracked poses for this video
        # Format: { track_id: [ {frame_id: int, keypoints: [[x,y,conf],...]} ] }
        video_tracks = {}
        
        for frame_idx, frame_path in enumerate(frames):
            img = cv2.imread(str(frame_path))
            
            # Run tracking inference
            # persist=True keeps IDs across frames. verbose=False hides per-frame logs.
            results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False, device=0)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints_batch = results[0].keypoints.data.cpu().numpy() # Shape: (N, 17, 3)
                
                for i, track_id in enumerate(track_ids):
                    if track_id not in video_tracks:
                        video_tracks[track_id] = []
                    
                    # Convert to standard Python float for JSON serialization
                    kpts = keypoints_batch[i].tolist()
                    video_tracks[track_id].append({
                        "frame_id": frame_idx,
                        "keypoints": kpts
                    })
            
            # Option A: Visualization
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
        out.release()
        
        # Save tracks to JSON
        json_path = pose_dir / f"{video_folder.name}_tracks.json"
        with open(json_path, 'w') as f:
            json.dump(video_tracks, f)

    print(f"\nProcessing finished for test subset.")
    print(f"Pose Data saved to: {POSE_OUTPUT_DIR}")
    print(f"Visualizations saved to: {VIS_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
