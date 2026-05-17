import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Constants for track quality and stability
MIN_TRACK_LENGTH = 12
MAX_INTERPOLATION_GAP = 2  # Only interpolate tiny gaps
MAX_GAP_THRESHOLD = 5      # Terminate track if gap is larger than this
MIN_AVG_MOTION = 1.0       # Minimum average pixels moved per frame to be considered "human"
CONF_THRESH = 0.2

def process_video_directory(video_dir, output_dir, model, tracker_config, limit=None):
    """Processes all videos in a directory, detecting and tracking people."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    video_files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
    
    if limit:
        video_files = video_files[:limit]
        print(f"Limiting to first {limit} videos in {video_dir}")

    for video_file in tqdm(video_files, desc=f"Processing {os.path.basename(video_dir)}"):
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_path = os.path.join(output_dir, f"{video_name}.json")
        
        if os.path.exists(output_path):
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video {video_path}")
            continue

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            original_fps = 30.0 # Standard fallback
            
        raw_tracks = {} # track_id -> list of detections
        frame_idx = 0
        
        # 1. Continuous Tracking on EVERY frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame is None:
                frame_idx += 1
                continue

            # Run YOLOv8 tracking with persist=True always
            results = model.track(
                source=frame,
                persist=True,
                classes=[0],
                tracker=tracker_config,
                conf=CONF_THRESH,
                verbose=False
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, ids, confs):
                    tid_str = str(track_id)
                    if tid_str not in raw_tracks:
                        raw_tracks[tid_str] = []
                    
                    x1, y1, x2, y2 = [float(val) for val in box]
                    # Store center directly to avoid repeated calculations
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    raw_tracks[tid_str].append({
                        "frame_idx": frame_idx,
                        "bbox": [x1, y1, x2, y2],
                        "center": [cx, cy],
                        "conf": float(conf)
                    })
            
            frame_idx += 1
        
        cap.release()

        # --- 2. Post-Processing: Sorting, Filtering & Micro-Interpolation ---
        final_tracks = {}
        for tid_str, data in raw_tracks.items():
            # 2.1 Fix track ordering bug (IMPORTANT)
            data.sort(key=lambda x: x["frame_idx"])
            
            if len(data) < MIN_TRACK_LENGTH:
                continue
            
            # 2.2 Motion-based filtering (Discard static false positives)
            centers = np.array([f["center"] for f in data])
            diffs = np.diff(centers, axis=0)
            motion_per_frame = np.sqrt(np.sum(diffs**2, axis=1))
            avg_motion = np.mean(motion_per_frame)
            
            if avg_motion < MIN_AVG_MOTION:
                continue # Discard static/noisy track

            refined_data = []
            for i in range(len(data) - 1):
                curr_f = data[i]
                next_f = data[i+1]
                
                # Add current frame (with calculated timestamp)
                curr_f["timestamp"] = round(curr_f["frame_idx"] / original_fps, 4)
                refined_data.append(curr_f)
                
                # 2.3 Frame-based interpolation (CRITICAL)
                gap = next_f["frame_idx"] - curr_f["frame_idx"]
                
                # Long-gap corruption handling: Terminate if gap > 5 frames
                if gap > MAX_GAP_THRESHOLD:
                    # In this simplified implementation, we just stop this segment.
                    # A more complex one would split into separate tracks.
                    break 
                
                # Micro-interpolation for gaps <= 2 frames
                if 1 < gap <= (MAX_INTERPOLATION_GAP + 1):
                    for j in range(1, gap):
                        alpha = j / gap
                        interp_idx = curr_f["frame_idx"] + j
                        interp_ts = round(interp_idx / original_fps, 4)
                        
                        interp_bbox = [curr_f["bbox"][k]*(1-alpha) + next_f["bbox"][k]*alpha for k in range(4)]
                        interp_center = [curr_f["center"][k]*(1-alpha) + next_f["center"][k]*alpha for k in range(2)]
                        interp_conf = curr_f["conf"]*(1-alpha) + next_f["conf"]*alpha
                        
                        refined_data.append({
                            "frame_idx": interp_idx,
                            "timestamp": interp_ts,
                            "bbox": interp_bbox,
                            "center": interp_center,
                            "conf": interp_conf
                        })
            
            # Add last frame if we didn't break early
            if len(refined_data) > 0 and refined_data[-1]["frame_idx"] < data[-1]["frame_idx"]:
                last_f = data[-1]
                last_f["timestamp"] = round(last_f["frame_idx"] / original_fps, 4)
                refined_data.append(last_f)
            
            if len(refined_data) >= MIN_TRACK_LENGTH:
                final_tracks[tid_str] = refined_data

        # 3. Save simplified output structure
        with open(output_path, 'w') as f:
            json.dump({
                "video_name": video_name, 
                "original_fps": original_fps,
                "tracks": final_tracks
            }, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Detect and track people (Clean temporal stream).")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--tracker", type=str, default="bytetrack_12fps.yaml", help="Tracker config file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos per category")
    
    args = parser.parse_args()

    if not os.path.exists(args.tracker):
        print(f"Error: Tracker config {args.tracker} not found.")
        exit(1)

    model = YOLO(args.model)
    
    for category in ["Violence", "NonViolence"]:
        print(f"\n--- Processing {category} Videos ---")
        process_video_directory(
            video_dir=f"data/videos/{category}",
            output_dir=f"data/tracks/{category}",
            model=model,
            tracker_config=args.tracker,
            limit=args.limit
        )
    
    print("\nStage 1: Detection and Tracking complete.")
