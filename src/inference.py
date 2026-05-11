import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import argparse
import time
from pathlib import Path

# Configuration
MODEL_YOLO = "yolov8n-pose.pt"
MODEL_GRU = "output/models/gru_model.keras"
TARGET_FPS = 5
SEQUENCE_LENGTH = 16
FIGHT_THRESHOLD = 0.5
SMOOTHING_WINDOW = 5
SMOOTHING_THRESHOLD = 3  # Trigger alert if 3/5 predictions are positive
INTERACTION_DISTANCE = 300  # Distance in pixels to gate GRU inference
MAX_LOST_FRAMES = 10

def get_center(keypoints):
    """Calculates the center of a person based on their 17 keypoints."""
    # Filter points with reasonable confidence
    valid_points = keypoints[keypoints[:, 2] > 0.1]
    if len(valid_points) == 0:
        return None
    return np.mean(valid_points[:, :2], axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (0) or path to video file")
    args = parser.parse_args()

    print("Loading models...")
    yolo = YOLO(MODEL_YOLO)
    gru = tf.keras.models.load_model(MODEL_GRU)
    
    # Track management
    # { track_id: {"buffer": deque(maxlen=16), "history": deque(maxlen=5), "lost": 0, "last_pose": None} }
    tracks = {}
    
    # Input Stream
    source = args.source if args.source != "0" else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    # Deterministic FPS skipping
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0: native_fps = 30 # Fallback for some webcams
    frame_skip = max(1, round(native_fps / TARGET_FPS))
    print(f"Native FPS: {native_fps:.2f} | Target FPS: {TARGET_FPS} | Skip Interval: {frame_skip}")

    # Setup Fullscreen Window
    win_name = "Fight Detection System - Live"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Step 1: YOLO-Pose Tracking
        # persist=True keeps IDs across frames.
        results = yolo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        current_frame_ids = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints_batch = results[0].keypoints.data.cpu().numpy() # (N, 17, 3)
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            
            for i, track_id in enumerate(track_ids):
                current_frame_ids.append(track_id)
                kpts = keypoints_batch[i] # (17, 3)
                
                if track_id not in tracks:
                    tracks[track_id] = {
                        "buffer": deque(maxlen=SEQUENCE_LENGTH),
                        "history": deque(maxlen=SMOOTHING_WINDOW),
                        "lost": 0,
                        "is_fight": False
                    }
                
                tracks[track_id]["buffer"].append(kpts)
                tracks[track_id]["lost"] = 0
                tracks[track_id]["last_pose"] = kpts
                tracks[track_id]["bbox"] = bboxes[i]

        # Step 2: Track Housekeeping (Remove lost tracks)
        all_track_ids = list(tracks.keys())
        for tid in all_track_ids:
            if tid not in current_frame_ids:
                tracks[tid]["lost"] += 1
                if tracks[tid]["lost"] > MAX_LOST_FRAMES:
                    del tracks[tid]

        # Step 3: Interaction Gating & GRU Inference
        # Batch GRU calls for all eligible tracks
        eligible_tids = []
        input_batch = []
        
        for tid, data in tracks.items():
            if len(data["buffer"]) == SEQUENCE_LENGTH:
                # Interaction Gate: Check distance to any OTHER active track
                center_a = get_center(data["last_pose"])
                if center_a is None: continue
                
                is_near_someone = False
                for other_tid, other_data in tracks.items():
                    if tid == other_tid: continue
                    center_b = get_center(other_data["last_pose"])
                    if center_b is None: continue
                    
                    dist = np.linalg.norm(center_a - center_b)
                    if dist < INTERACTION_DISTANCE:
                        is_near_someone = True
                        break
                
                if is_near_someone:
                    eligible_tids.append(tid)
                    # Flatten sequence (16, 17, 3) -> (16, 51)
                    seq = np.array(data["buffer"]).reshape(SEQUENCE_LENGTH, 51)
                    input_batch.append(seq)
                else:
                    # Not interacting? Clear history to prevent stale alerts
                    data["history"].append(0)
                    data["is_fight"] = False

        if input_batch:
            preds = gru.predict(np.array(input_batch), verbose=0)
            for i, tid in enumerate(eligible_tids):
                prob = preds[i][0]
                # Step 4: Temporal Smoothing (3/5 Rule)
                tracks[tid]["history"].append(1 if prob > FIGHT_THRESHOLD else 0)
                tracks[tid]["is_fight"] = (sum(tracks[tid]["history"]) >= SMOOTHING_THRESHOLD)

        # Step 5: Visual Feedback
        annotated_frame = frame.copy()
        
        # Draw bounding boxes and labels
        for tid, data in tracks.items():
            if "bbox" not in data: continue
            x1, y1, x2, y2 = map(int, data["bbox"])
            
            is_fight = data.get("is_fight", False)
            color = (0, 0, 255) if is_fight else (0, 255, 0)
            bg_color = (0, 0, 200) if is_fight else (0, 200, 0)
            text_color = (255, 255, 255)
            
            # Label includes the temporal history score
            history_sum = sum(data["history"]) if "history" in data else 0
            label = f"ID:{tid} FIGHT! ({history_sum}/5)" if is_fight else f"ID:{tid} OK ({history_sum}/5)"
            
            # Draw box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background for readability
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), bg_color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Global screen alert
        any_fight = any(d.get("is_fight", False) for d in tracks.values())
        if any_fight:
            # Draw a massive red banner at the top
            cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], 80), (0, 0, 255), -1)
            cv2.putText(annotated_frame, "!!! VIOLENCE DETECTED !!!", (50, 55), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)

        # System Overlay (Semi-transparent stats box at the bottom left)
        overlay = annotated_frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (10, h - 120), (370, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        
        cv2.putText(annotated_frame, "SYSTEM STATUS:", (20, h - 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Active Tracks: {len(tracks)}", (20, h - 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated_frame, f"GRU Calls (This Frame): {len(eligible_tids)}", (20, h - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated_frame, f"Sampling: {TARGET_FPS} FPS", (20, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(win_name, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
