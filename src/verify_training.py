import os
import shutil
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from train import train_model

def create_dummy_data(dummy_dir="dummy_data"):
    path = Path(dummy_dir)
    categories = ["Violence", "NonViolence"]
    splits = ["train", "val"]
    
    for s in splits:
        for c in categories:
            d = path / s / c / "video_1"
            d.mkdir(parents=True, exist_ok=True)
            # Create 2 dummy images per split/category
            for i in range(2):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                cv2.imwrite(str(d / f"frame_{i}.jpg"), img)
    return dummy_dir

def verify():
    print("Creating dummy dataset for fast verification...")
    dummy_dir = create_dummy_data()
    
    print("Starting verification test...")
    try:
        # Run training on dummy data
        train_model(data_dir=dummy_dir, batch_size=2, epochs_phase1=1, epochs_phase2=1)
        print("\nVerification test passed!")
    except Exception as e:
        print(f"\nVerification test failed: {e}")
    finally:
        if os.path.exists(dummy_dir):
            shutil.rmtree(dummy_dir)

if __name__ == "__main__":
    verify()
