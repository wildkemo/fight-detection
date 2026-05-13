import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import argparse
import time
import threading
from pathlib import Path
from queue import Queue

# Configuration
MODEL_YOLO = "yolov8s-pose.pt"
MODEL_TCN = "output/models/tcn_model_quant.tflite"
TARGET_FPS = 12
FRAME_INTERVAL = 1.0 / TARGET_FPS
SEQUENCE_LENGTH = 36
FIGHT_THRESHOLD = 0.75
SMOOTHING_WINDOW = 8
SMOOTHING_THRESHOLD = 6
INTERACTION_DISTANCE = 150
MAX_LOST_FRAMES = 36
PROCESSING_WIDTH = 640   # Resize to this width for YOLO (keeps aspect ratio)
PROCESSING_HEIGHT = 360  # Optimized 16:9 resolution

class AsyncTCNPredictor:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get quantization parameters
        self.in_scale, self.in_zero_point = self.input_details[0]['quantization']
        self.out_scale, self.out_zero_point = self.output_details[0]['quantization']
        
        self.input_queue = Queue(maxsize=5)
        self.output_store = {}
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._predict_loop, daemon=True)
        self.thread.start()
    
    def _predict_loop(self):
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.05)
                if item is None:
                    break
                batch_id, input_batch = item
                if len(input_batch) > 0:
                    results = []
                    for seq in input_batch:
                        # Quantize: float32 -> int8
                        q_input = (seq / self.in_scale + self.in_zero_point).astype(np.int8)
                        q_input = np.expand_dims(q_input, axis=0)
                        
                        self.interpreter.set_tensor(self.input_details[0]['index'], q_input)
                        self.interpreter.invoke()
                        
                        # Dequantize: int8 -> float32
                        q_output = self.interpreter.get_tensor(self.output_details[0]['index'])
                        prob = (q_output.astype(np.float32) - self.out_zero_point) * self.out_scale
                        results.append(prob[0])
                    
                    with self.lock:
                        self.output_store[batch_id] = np.array(results)
            except Exception as e:
                # print(f"Predict loop error: {e}")
                continue
    
    def predict_async(self, batch_id, input_batch):
        if not self.input_queue.full():
            self.input_queue.put((batch_id, input_batch))
            return True
        return False
    
    def get_result(self, batch_id):
        with self.lock:
            return self.output_store.pop(batch_id, None)
    
    def stop(self):
        self.running = False
        self.input_queue.put(None)

class FastFrameGrabber:
    def __init__(self, source):
        if isinstance(source, int) or source == "0":
            self.cap = cv2.VideoCapture(int(source))
        else:
            self.cap = cv2.VideoCapture(source)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
    
    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Store frame at original size for display
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.001)
    
    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None
    
    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()

def get_center(keypoints):
    valid = keypoints[keypoints[:, 2] > 0.1]
    if len(valid) == 0:
        return None
    return np.mean(valid[:, :2], axis=0)

def fast_preprocess(frame):
    """Light preprocessing only when needed (night mode)"""
    # Skip heavy denoise by default, just return as-is
    # CLAHE is also skipped unless night_mode is enabled
    return frame

def night_preprocess(frame):
    """Faster preprocessing for low light"""
    # Fast bilateral filter for edge-preserving smoothing
    frame = cv2.bilateralFilter(frame, 5, 50, 50)
    # CLAHE on L channel in LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--night-mode", action="store_true", 
                       help="Enable heavy preprocessing (reduces FPS)")
    parser.add_argument("--resolution", type=str, default="640x360",
                       help="Processing resolution WxH (e.g., 640x360, 854x480)")
    args = parser.parse_args()

    # Parse resolution
    try:
        res_w, res_h = map(int, args.resolution.split('x'))
        global PROCESSING_WIDTH, PROCESSING_HEIGHT
        PROCESSING_WIDTH, PROCESSING_HEIGHT = res_w, res_h
    except:
        print(f"Invalid resolution {args.resolution}, using 640x360")
    
    print(f"Processing at {PROCESSING_WIDTH}x{PROCESSING_HEIGHT}")

    print("Loading models...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        device = 0
        print(f"GPU: {gpus[0]}")
    else:
        device = 'cpu'
        print("CPU mode")
    
    yolo = YOLO(MODEL_YOLO)
    if device == 0:
        yolo.to('cuda')
    
    tcn = AsyncTCNPredictor(MODEL_TCN)
    
    source = int(args.source) if args.source == "0" else args.source
    grabber = FastFrameGrabber(source)
    
    if not grabber.cap.isOpened():
        print(f"Error: Cannot open {source}")
        return
    
    # Choose preprocessing function
    preprocess_func = night_preprocess if args.night_mode else fast_preprocess
    
    tracks = {}
    pending_tcn = {}
    next_batch_id = 0
    
    # Wait for first frame to get native resolution
    print("Waiting for stream...")
    first_frame = None
    while first_frame is None:
        ret, first_frame = grabber.get_frame()
        if not ret:
            time.sleep(0.1)
            continue
    
    h_orig, w_orig = first_frame.shape[:2]
    # Force 16:9 resolution for the entire pipeline
    preview_w = w_orig
    preview_h = int(w_orig * 9 / 16)
    print(f"Stream detected: {w_orig}x{h_orig} -> Forced 16:9: {preview_w}x{preview_h}")

    # Setup window with forced 16:9 aspect ratio
    cv2.namedWindow("Fight Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fight Detection", preview_w, preview_h)
    
    # Recording
    output_dir = Path("output/detected_fights")
    output_dir.mkdir(parents=True, exist_ok=True)
    is_recording = False
    video_writer = None
    cooldown_counter = 0
    
    print(f"System running at {TARGET_FPS} FPS (target)")
    print("Press 'q' to quit\n")
    
    # Timing
    next_frame_time = time.time()
    frame_count = 0
    fps_display = 0
    last_fps_time = time.time()
    
    # Initial frame resize and setup
    first_frame = cv2.resize(first_frame, (preview_w, preview_h))
    last_frame = first_frame # Reuse the frame we already grabbed
    
    while True:
        current_time = time.time()
        
        # Strict 12 FPS timing
        if current_time < next_frame_time:
            sleep_time = next_frame_time - current_time
            if sleep_time > 0.002:
                time.sleep(sleep_time * 0.5)
            continue
        
        next_frame_time = current_time + FRAME_INTERVAL
        
        # Get frame
        ret, frame = grabber.get_frame()
        if not ret or frame is None:
            if last_frame is not None:
                frame = last_frame
            else:
                continue
        else:
            # Force 16:9 aspect ratio
            frame = cv2.resize(frame, (preview_w, preview_h))
            last_frame = frame
        
        frame_count += 1
        
        # Resize for YOLO processing (maintains aspect ratio)
        h, w = frame.shape[:2]
        scale = min(PROCESSING_WIDTH / w, PROCESSING_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        scale_w = new_w / w
        scale_h = new_h / h
        proc_frame = cv2.resize(frame, (new_w, new_h))
        
        # Apply optional preprocessing
        proc_frame = preprocess_func(proc_frame)
        
        # YOLO tracking on resized frame
        results = yolo.track(proc_frame, 
                            persist=True,
                            tracker="bytetrack.yaml",
                            conf=0.25,
                            iou=0.5,
                            verbose=False,
                            device=device)
        
        # Scale boxes back to original frame coordinates
        current_track_ids = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            keypoints = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else None
            
            # Scale boxes and keypoints back to original frame size
            for i, track_id in enumerate(track_ids):
                current_track_ids.append(track_id)
                
                # Scale box
                x1, y1, x2, y2 = boxes[i]
                x1 = int(x1 / scale_w)
                x2 = int(x2 / scale_w)
                y1 = int(y1 / scale_h)
                y2 = int(y2 / scale_h)
                
                # Scale keypoints
                if keypoints is not None and i < len(keypoints):
                    kpts = keypoints[i].copy()
                    kpts[:, 0] /= scale_w
                    kpts[:, 1] /= scale_h
                else:
                    kpts = None
                
                # Initialize or update track
                if track_id not in tracks:
                    tracks[track_id] = {
                        "buffer": deque(maxlen=SEQUENCE_LENGTH),
                        "history": deque(maxlen=SMOOTHING_WINDOW),
                        "lost_frames": 0,
                        "is_fight": False,
                        "last_prob": 0.0,
                        "bbox": (x1, y1, x2, y2),
                        "last_center": None
                    }
                else:
                    tracks[track_id]["bbox"] = (x1, y1, x2, y2)
                
                tracks[track_id]["lost_frames"] = 0
                if kpts is not None:
                    tracks[track_id]["buffer"].append(kpts)
                    tracks[track_id]["last_center"] = get_center(kpts)
        
        # Remove lost tracks
        for track_id in list(tracks.keys()):
            if track_id not in current_track_ids:
                tracks[track_id]["lost_frames"] += 1
                if tracks[track_id]["lost_frames"] > MAX_LOST_FRAMES:
                    del tracks[track_id]
        
        # Process completed TCN predictions
        for batch_id in list(pending_tcn.keys()):
            result = tcn.get_result(batch_id)
            if result is not None:
                track_ids = pending_tcn[batch_id]
                for i, track_id in enumerate(track_ids):
                    if track_id in tracks:
                        prob = result[i][0]
                        tracks[track_id]["last_prob"] = prob
                        tracks[track_id]["history"].append(1 if prob > FIGHT_THRESHOLD else 0)
                        tracks[track_id]["is_fight"] = sum(tracks[track_id]["history"]) >= SMOOTHING_THRESHOLD
                del pending_tcn[batch_id]
        
        # Interaction detection and TCN submission
        eligible_batch = []
        track_items = list(tracks.items())
        
        for track_id, data in track_items:
            if len(data["buffer"]) == SEQUENCE_LENGTH and data["last_center"] is not None:
                is_interacting = False
                for other_id, other_data in track_items:
                    if track_id == other_id or other_data["last_center"] is None:
                        continue
                    dist = np.linalg.norm(data["last_center"] - other_data["last_center"])
                    if dist < INTERACTION_DISTANCE:
                        is_interacting = True
                        break
                
                if is_interacting:
                    eligible_batch.append(track_id)
                else:
                    data["history"].append(0)
                    data["is_fight"] = False
        
        if eligible_batch:
            batch_input = []
            for track_id in eligible_batch:
                seq = np.array(tracks[track_id]["buffer"]).reshape(SEQUENCE_LENGTH, 51)
                batch_input.append(seq)
            if batch_input:
                batch_id = next_batch_id
                next_batch_id += 1
                tcn.predict_async(batch_id, np.array(batch_input))
                pending_tcn[batch_id] = eligible_batch
        
        # Visualization on original frame (already a copy from grabber)
        display = frame
        
        # Calculate how many active tracks are fighting (mutual combat requirement)
        fighting_tracks = [track_id for track_id, data in track_items if data["is_fight"] and data["bbox"] is not None and data["lost_frames"] <= 2]
        any_fight = len(fighting_tracks) >= 2
        
        for track_id, data in track_items:
            if data["bbox"] is None or data["lost_frames"] > 2:
                continue
            x1, y1, x2, y2 = data["bbox"]
            is_fight = data["is_fight"]
            color = (0, 0, 255) if is_fight else (0, 255, 0)
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id}"
            if is_fight:
                label += " FIGHT!"
            elif data["last_prob"] > 0:
                label += f" {data['last_prob']*100:.0f}%"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        if any_fight:
            cv2.rectangle(display, (0, 0), (display.shape[1], 70), (0,0,255), -1)
            cv2.putText(display, "!!! VIOLENCE DETECTED !!!", (display.shape[1]//2 - 200, 45),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2)
        
        # Recording
        if any_fight:
            if not is_recording:
                is_recording = True
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                out_path = output_dir / f"fight_{timestamp}.mp4"
                h, w = display.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(out_path), fourcc, TARGET_FPS, (w, h))
                print(f"[REC] Started: {out_path.name}")
            cooldown_counter = 0
        elif is_recording:
            cooldown_counter += 1
            if cooldown_counter > TARGET_FPS * 3:
                is_recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print("[REC] Stopped")
        
        if is_recording and video_writer:
            video_writer.write(display)
        
        # FPS display
        if current_time - last_fps_time >= 1.0:
            fps_display = frame_count
            frame_count = 0
            last_fps_time = current_time
        
        # Status overlay
        cv2.putText(display, f"FPS: {fps_display} | Tracks: {len(tracks)} | Queue: {len(pending_tcn)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # Show window - it will maintain original frame aspect ratio
        cv2.imshow("Fight Detection", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    tcn.stop()
    grabber.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("\nStopped")

if __name__ == "__main__":
    main()