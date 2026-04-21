import numpy as np
import cv2

def dynamic_range(frame):
    """
    Step 4: Contrast Stretching (Normalization).
    
    Linearly scales the intensity levels of the image to occupy the 
    full dynamic range [0, 255].
    
    Formula: P_out = (P_in - min) * (255 / (max - min))
    """
    # Convert to float for precise calculation
    frame_float = frame.astype(np.float32)
    
    min_val = np.min(frame_float)
    max_val = np.max(frame_float)
    
    # Avoid division by zero for uniform images
    if max_val == min_val:
        return frame
    
    # Apply stretching formula
    stretched = (frame_float - min_val) * (255.0 / (max_val - min_val))
    
    # Clip just in case of rounding errors and convert back to uint8
    return np.clip(stretched, 0, 255).astype(np.uint8)
