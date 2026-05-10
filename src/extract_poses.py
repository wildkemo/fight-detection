import cv2
import numpy as np
import os
import json
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_DIR = Path("output/crops")
OUTPUT_DIR = Path("output/poses")
EVAL_DIR = Path("output/eval")
# MoveNet Lightning from TF Hub
MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
IMG_SIZE = 192
CONF_THRESHOLD = 0.2

def letterbox(img, new_shape=(192, 192), color=(114, 114, 114)):
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
    
    return img

def main():
    print("Loading MoveNet Lightning model from TF Hub...")
    # hub.load handles the download and loading of the model
    module = hub.load(MODEL_URL)
    movenet = module.signatures['serving_default']
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "total_poses_extracted": 0,
        "total_joints_processed": 0,
        "low_confidence_joints": 0,
        "video_metrics": {}
    }

    # Get all video folders in input_dir
    video_folders = list(INPUT_DIR.glob("**/*/"))
    video_folders = [f for f in video_folders if any(f.glob("*.jpg"))]
    
    print(f"Found {len(video_folders)} videos to process.")

    for video_folder in tqdm(video_folders, desc="Extracting poses"):
        rel_path = video_folder.relative_to(INPUT_DIR)
        output_video_dir = OUTPUT_DIR / rel_path
        output_video_dir.mkdir(parents=True, exist_ok=True)
        
        crop_files = sorted(list(video_folder.glob("*.jpg")))
        video_conf_sum = 0
        video_low_conf_count = 0
        video_joints_count = 0
        
        for crop_path in crop_files:
            img = cv2.imread(str(crop_path))
            if img is None: continue
            
            # Step 1: Preprocess (Letterbox to 192x192)
            padded_img = letterbox(img, IMG_SIZE)
            input_image = tf.expand_dims(padded_img, axis=0)
            input_image = tf.cast(input_image, dtype=tf.int32)
            
            # Step 2: MoveNet Inference
            outputs = movenet(input_image)
            # Output shape: [1, 1, 17, 3] -> [y, x, score]
            keypoints_with_scores = outputs['output_0'].numpy()[0, 0, :, :]
            
            # Step 3: Save Pose
            pose_name = output_video_dir / f"{crop_path.stem}_pose.npy"
            np.save(pose_name, keypoints_with_scores)
            
            # Step 4: Component-Level Metrics
            metrics["total_poses_extracted"] += 1
            scores = keypoints_with_scores[:, 2]
            
            video_conf_sum += float(np.sum(scores))
            video_joints_count += int(len(scores))
            
            low_conf = int(np.sum(scores < CONF_THRESHOLD))
            video_low_conf_count += low_conf
            metrics["low_confidence_joints"] += low_conf
            metrics["total_joints_processed"] += int(len(scores))
        
        metrics["video_metrics"][str(rel_path)] = {
            "total_poses": int(len(crop_files)),
            "avg_confidence": float(video_conf_sum / video_joints_count) if video_joints_count > 0 else 0.0,
            "low_confidence_rate": float(video_low_conf_count / video_joints_count) if video_joints_count > 0 else 0.0
        }

    # Final summary
    if metrics["total_joints_processed"] > 0:
        metrics["overall_avg_confidence"] = float(np.sum([v["avg_confidence"] for v in metrics["video_metrics"].values()]) / len(metrics["video_metrics"]))
        metrics["overall_low_confidence_rate"] = metrics["low_confidence_joints"] / metrics["total_joints_processed"]
    
    with open(EVAL_DIR / "movenet_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nProcessing finished for test subset.")
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"Metrics saved to {EVAL_DIR / 'movenet_metrics.json'}")

if __name__ == "__main__":
    main()
