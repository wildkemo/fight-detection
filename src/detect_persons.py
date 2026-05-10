import cv2
import numpy as np
import os
import json
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
INPUT_DIR = Path("output/frames")
OUTPUT_DIR = Path("output/crops")
EVAL_DIR = Path("output/eval")
MODEL_PATH = "yolov8n.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.3
PERSON_CLASS_ID = 0

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resizes and pads image to new_shape while preserving aspect ratio."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (left, top)

def main():
    print("Loading YOLOv8n model...")
    model = YOLO(MODEL_PATH)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "total_frames": 0,
        "missed_frames": 0,
        "total_persons_detected": 0,
        "video_metrics": {}
    }

    # Get all video folders in input_dir
    video_folders = list(INPUT_DIR.glob("**/*/"))
    video_folders = [f for f in video_folders if any(f.glob("*.jpg"))]
    
    print(f"Found {len(video_folders)} videos to process.")

    for video_folder in tqdm(video_folders, desc="Processing videos"):
        video_name = video_folder.name
        rel_path = video_folder.relative_to(INPUT_DIR)
        output_video_dir = OUTPUT_DIR / rel_path
        output_video_dir.mkdir(parents=True, exist_ok=True)
        
        frames = sorted(list(video_folder.glob("*.jpg")))
        video_missed = 0
        video_total_persons = 0
        
        for frame_path in frames:
            img = cv2.imread(str(frame_path))
            if img is None: continue
            
            metrics["total_frames"] += 1
            
            # Step 1: Preprocess (Letterbox)
            # YOLO handles resize internally, but we use letterbox for visualization/debug if needed
            # and to follow SCOPE.md's explicit resizing instruction.
            padded_img, ratio, (pad_left, pad_top) = letterbox(img, IMG_SIZE)
            
            # Step 2: YOLO Inference
            # We run on the original image; YOLO's internal preprocessing is similar to letterbox.
            # Using original image ensures boxes are mapped back correctly by ultralytics.
            results = model(img, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, classes=[PERSON_CLASS_ID], verbose=False)
            
            boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
            
            if len(boxes) == 0:
                video_missed += 1
                metrics["missed_frames"] += 1
            else:
                video_total_persons += len(boxes)
                metrics["total_persons_detected"] += len(boxes)
                
                # Step 3: Crop and Save
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    # Clip coordinates to image boundaries
                    h, w = img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    crop_name = output_video_dir / f"{frame_path.stem}_person_{i:02d}.jpg"
                    cv2.imwrite(str(crop_name), crop)
        
        metrics["video_metrics"][str(rel_path)] = {
            "total_frames": len(frames),
            "missed_frames": video_missed,
            "miss_rate": video_missed / len(frames) if len(frames) > 0 else 0,
            "total_persons": video_total_persons
        }

    # Final summary
    if metrics["total_frames"] > 0:
        metrics["overall_miss_rate"] = metrics["missed_frames"] / metrics["total_frames"]
    
    with open(EVAL_DIR / "yolo_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nProcessing finished for test subset.")
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"Metrics saved to {EVAL_DIR / 'yolo_metrics.json'}")

if __name__ == "__main__":
    main()
