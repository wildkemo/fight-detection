import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import sys
import os

class LiveFightDetector:
    def __init__(self, model_path="models/fight_detection.tflite", window_size=32, threshold=0.7, fps_target=5, 
                 motion_threshold=1.0, pixel_diff_threshold=25):
        print(f"Loading TFLite model from {model_path}...")
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            print("Please ensure you have completed training and generated the .tflite file.")
            sys.exit(1)
            
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.window_size = window_size
        self.threshold = threshold
        self.fps_target = fps_target
        self.prediction_buffer = deque(maxlen=window_size)
        
        # Motion Filter State
        self.prev_gray = None
        self.motion_threshold = motion_threshold  # % of pixels that must change
        self.pixel_diff_threshold = pixel_diff_threshold # Threshold for pixel difference

    def check_motion(self, frame):
        """
        Fast frame differencing to detect if significant motion is present.
        Returns True if motion is detected, False otherwise.
        """
        # 1. Grayscale and Blur (Downscale for speed)
        small_frame = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return True # Assume motion on first frame to initialize

        # 2. Compute Absolute Difference
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, self.pixel_diff_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # 3. Calculate % of pixels changed
        changed_pixels = np.sum(thresh > 0)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_percent = (changed_pixels / total_pixels) * 100
        
        self.prev_gray = gray
        
        return motion_percent > self.motion_threshold

    def fast_preprocess(self, frame):
        """
        Optimized preprocessing using OpenCV built-ins for real-time performance.
        """
        # 1. Adaptive Resize with Padding (Built-in + Manual Pad)
        h, w = frame.shape[:2]
        target_h, target_w = 224, 224
        
        aspect = w / h
        if aspect > (target_w / target_h): # Wide
            new_w = target_w
            new_h = int(target_w / aspect)
        else: # Tall
            new_h = target_h
            new_w = int(target_h * aspect)
            
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad with black to 224x224
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        # 2. Denoising (Built-in Gaussian) - DISABLED
        # denoised = cv2.GaussianBlur(canvas, (3, 3), 0)
        
        # 3. CLAHE (Built-in on Y channel of YCrCb to match training logic) - DISABLED
        # ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
        # y, cr, cb = cv2.split(ycrcb)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # cl = clahe.apply(y)
        # ycrcb_img = cv2.merge((cl, cr, cb))
        # processed_bgr = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        
        # 4. Color Space Conversion (BGR to RGB)
        # The model was trained using image_dataset_from_directory which defaults to RGB
        processed_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # 5. Normalization: REMOVED manual /255.0
        # EfficientNetB0 has a built-in Rescaling(1/255) layer.
        # Passing [0, 1] data would result in the model seeing [0, 1/255].
        final_frame = processed_rgb.astype(np.float32)
        
        # Add batch dimension: (1, 224, 224, 3)
        return np.expand_dims(final_frame, axis=0)

    def predict(self, frame):
        """
        Runs inference and applies sliding window aggregation.
        Includes a motion-gating mechanism to reduce false positives on static scenes.
        """
        # 1. Motion Gating
        if not self.check_motion(frame):
            # No significant motion, decay the confidence to 0
            self.prediction_buffer.append(0.0)
            avg_prob = np.mean(self.prediction_buffer)
            return avg_prob >= self.threshold, avg_prob

        # 2. Heavy CNN Inference
        input_data = self.fast_preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Output is logit (from BCEWithLogitsLoss)
        logit = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        
        # Sigmoid to get probability
        probability = 1.0 / (1.0 + np.exp(-logit))
        
        self.prediction_buffer.append(probability)
        avg_prob = np.mean(self.prediction_buffer)
        
        return avg_prob >= self.threshold, avg_prob

def run_live():
    detector = LiveFightDetector()
    
    # Open webcam (0 is default)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    print("--- Fight Detection Live Active ---")
    print("Throttled to 5 FPS. Press 'q' to quit.")
    
    target_delay = 1.0 / detector.fps_target
    prev_time = 0
    
    while True:
        curr_time = time.time()
        # Ensure we only process at the target FPS (5 FPS)
        if (curr_time - prev_time) >= target_delay:
            prev_time = curr_time
            
            ret, frame = cap.read()
            if not ret: 
                print("Failed to capture frame.")
                break
            
            is_fight, confidence = detector.predict(frame)
            
            # UI Visualization
            color = (0, 0, 255) if is_fight else (0, 255, 0) # Red for Fight, Green for Normal
            label = "!!! FIGHT DETECTED !!!" if is_fight else "Normal Activity"
            
            # UI Overlay
            # 1. Outer Border
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 15)
            
            # 2. Status Label
            cv2.putText(frame, f"{label}", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
            
            # 3. Confidence Bar / Text
            cv2.putText(frame, f"Aggregated Confidence: {confidence*100:.1f}%", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 4. Buffer Status
            cv2.putText(frame, f"Window: {len(detector.prediction_buffer)}/16 frames", (20, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Fight Detection System - LIVE", frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Live inference terminated.")

if __name__ == "__main__":
    run_live()
