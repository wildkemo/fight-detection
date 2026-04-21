import cv2
import numpy as np

def log_transform(frame):
    """
    Step 10: Logarithmic Transformation.
    Expands the values of dark pixels in the image.
    """
    frame_float = frame.astype(np.float32)
    
    # We use a slight shift to ensure the curve is steep in dark regions
    # s = c * log(1 + r)
    c = 255 / np.log(1 + 255)
    log_image = c * np.log(1 + frame_float)
    
    return log_image.astype(np.uint8)
