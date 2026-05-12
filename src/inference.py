import os
# Suppress TensorFlow and CUDA logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import argparse
import time
import threading
from pathlib import Path
from src.preprocessors.denoiser import denoise_frame
from src.preprocessors.clahe import apply_clahe
from src.preprocessors.normalizer import normalize_frame, denormalize_to_uint8

# Configuration
MODEL_YOLO = "yolov8s-pose.pt"
MODEL_GRU = "output/models/gru_model.keras"
TARGET_FPS = 30
SEQUENCE_LENGTH = 96
FIGHT_THRESHOLD = 0.5
SMOOTHING_WINDOW = 15
SMOOTHING_THRESHOLD = 9  # Trigger alert if 9/15 predictions are positive
INTERACTION_DISTANCE = 300  # Distance in pixels to gate GRU inference
MAX_LOST_FRAMES = 30
MAX_FRAME_AGE = 2.0  # Seconds to tolerate CCTV network jitter

class ThreadedFrameGrabber:
    """
    Threaded background frame reader that continuously empties the camera 
    buffer to prevent RTSP latency accumulation.
    """
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        # Attempt to disable double buffering to reduce RTSP latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ret = False
        self.latest_frame = None
        self.timestamp = 0
        self.lock = threading.Lock()
        self.running = True
        
        # Start background thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            
            # MUST copy frame to prevent OpenCV internal buffer reuse from corrupting sequences
            with self.lock:
                self.ret = ret
                self.latest_frame = frame.copy()
                self.timestamp = time.time()

    def get_latest(self):
        with self.lock:
            return self.ret, self.latest_frame, self.timestamp

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def get_center(keypoints):
    """Calculates the center of a person based on their 17 keypoints."""
    valid_points = keypoints[keypoints[:, 2] > 0.1]
    if len(valid_points) == 0:
        return None
    return np.mean(valid_points[:, :2], axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (0) or RTSP/video path")
    parser.add_argument("--night-mode", action="store_true", help="Enable contrast enhancement for night/grayscale video")
    args = parser.parse_args()

    print("Loading models...")
    yolo = YOLO(MODEL_YOLO)
    gru = tf.keras.models.load_model(MODEL_GRU)
    
    tracks = {}
    
    # Input Stream via Background Thread
    source = args.source if args.source != "0" else 0
    grabber = ThreadedFrameGrabber(source)
    
    if not grabber.cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    # Setup Fullscreen Window with Aspect Ratio preservation
    win_name = "Fight Detection System - Live"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Fixed-Step Scheduler (Accumulator-based game loop style)
    step_interval = 1.0 / TARGET_FPS
    next_process_time = time.time()
    
    last_valid_frame = None
    
    # Recording setup
    RECORD_OUTPUT_DIR = Path("output/detected_fights")
    RECORD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    is_recording = False
    video_writer = None
    frames_since_fight = 0
    RECORD_COOLDOWN_FRAMES = 36  # 3 seconds of cooldown at 12 FPS

    print(f"Inference started. Target FPS: {TARGET_FPS}")

    while grabber.running:
        current_time = time.time()
        
        # Fixed-step accumulation to prevent temporal drift
        if current_time < next_process_time:
            # Yield CPU while waiting for the next logical frame tick
            time.sleep(0.001)
            continue
            
        next_process_time += step_interval

        # Pull latest frame + metadata
        ret, frame, frame_ts = grabber.get_latest()
        
        # Frame Freshness Validation & Fallback
        is_stale = False
        if not ret or frame is None:
            if last_valid_frame is not None:
                frame = last_valid_frame
                is_stale = True
            else:
                continue
        else:
            # Check age (Soft rejection threshold to handle CCTV jitter)
            age = current_time - frame_ts
            if age > MAX_FRAME_AGE:
                # Severe network lag: Re-use last valid but mark as stale
                if last_valid_frame is not None:
                    frame = last_valid_frame
                    is_stale = True
                else:
                    continue
            else:
                last_valid_frame = frame # Update fallback

        # 3-Step Preprocessing Pipeline (Align with Training)
        # 1. Denoise -> 2. CLAHE -> 3. Normalize/Denormalize (Clipping)
        proc_frame = denoise_frame(frame)
        proc_frame = apply_clahe(proc_frame)
        proc_frame = normalize_frame(proc_frame)
        proc_frame = denormalize_to_uint8(proc_frame)

        # Step 1: YOLO-Pose Tracking
        # Use slightly lower confidence threshold for live tracking stability
        conf_thresh = 0.25 
        results = yolo.track(proc_frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=conf_thresh)
        
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
                        "is_fight": False,
                        "last_prob": 0.0
                    }
                
                # Append to buffer. Stale frames are processed to maintain sequence continuity
                # but don't represent fresh motion.
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
                    # Flatten sequence (36, 17, 3) -> (36, 51)
                    seq = np.array(data["buffer"]).reshape(SEQUENCE_LENGTH, 51)
                    input_batch.append(seq)
                else:
                    # Not interacting? Clear history to prevent stale alerts
                    data["history"].append(0)
                    data["is_fight"] = False
                    data["last_prob"] = 0.0

        if input_batch:
            preds = gru.predict(np.array(input_batch), verbose=0)
            for i, tid in enumerate(eligible_tids):
                prob = preds[i][0]
                tracks[tid]["last_prob"] = prob
                # Step 4: Temporal Smoothing (4/6 Rule)
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
            
            # Label includes the temporal history score and probability percentage
            history_sum = sum(data["history"]) if "history" in data else 0
            prob_pct = data.get("last_prob", 0.0) * 100
            label = f"ID:{tid} FIGHT! {prob_pct:.1f}% ({history_sum}/6)" if is_fight else f"ID:{tid} OK {prob_pct:.1f}% ({history_sum}/6)"
            
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

        # Recording Logic
        if any_fight:
            if not is_recording:
                is_recording = True
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                out_path = RECORD_OUTPUT_DIR / f"fight_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(str(out_path), fourcc, TARGET_FPS, (w, h))
                print(f"🚨 Started recording: {out_path}")
            frames_since_fight = 0
        else:
            if is_recording:
                frames_since_fight += 1
                if frames_since_fight > RECORD_COOLDOWN_FRAMES:
                    is_recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    print("🛑 Stopped recording.")

        if is_recording and video_writer is not None:
            # Draw recording indicator
            cv2.circle(annotated_frame, (frame.shape[1] - 30, 40), 10, (0, 0, 255), -1)
            cv2.putText(annotated_frame, "REC", (frame.shape[1] - 80, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            video_writer.write(annotated_frame)

        # System Overlay with Stream Freshness Stats
        overlay = annotated_frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (10, h - 145), (370, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        
        cv2.putText(annotated_frame, "SYSTEM STATUS:", (20, h - 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Active Tracks: {len(tracks)}", (20, h - 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated_frame, f"GRU Calls: {len(eligible_tids)}", (20, h - 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        status_color = (0, 0, 255) if is_stale else (0, 255, 0)
        status_text = "DROPPING FRAMES / STALE" if is_stale else "STABLE (LIVE)"
        cv2.putText(annotated_frame, f"Stream Status: {status_text}", (20, h - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        cv2.putText(annotated_frame, f"Logic Rate: {TARGET_FPS} FPS", (20, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Force 16:9 Aspect Ratio for non-square DVR streams (e.g. 1080N / 960x1080)
        display_frame = cv2.resize(annotated_frame, (1280, 720))

        cv2.imshow(win_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if video_writer is not None:
        video_writer.release()
    grabber.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
)
