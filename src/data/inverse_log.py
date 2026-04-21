import cv2
import numpy as np

def inverse_log(frame):
    """
    Step 11: Inverse Log (Exponential) Transformation.
    Expands the values of bright pixels in the image.
    """
    # Normalize to [0, 1]
    frame_norm = frame.astype(np.float32) / 255.0
    
    # Use a power factor to make it more aggressive
    # Higher power = more expansion of highlights
    gamma = 2.0
    res = np.power(frame_norm, gamma)
    
    # Scale back
    inv_log_image = res * 255.0
    
    return inv_log_image.astype(np.uint8)
