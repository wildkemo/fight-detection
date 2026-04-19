import numpy as np
import cv2

def apply_dynamic_range_adjustment(frame):
    """
    Step 4: Apply Log or Inverse-Log transformation based on average brightness.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    # Constants for transformation
    c = 255 / np.log(1 + np.max(frame))
    
    if mean_brightness < 127:
        # Dark image -> Log Transformation to expand dark intensities
        log_image = c * (np.log(frame + 1))
        return np.array(log_image, dtype=np.uint8)
    else:
        # Bright image -> Inverse-Log (Exp) Transformation
        # Placeholder for inverse-log logic
        inv_log_image = np.exp(frame / c) - 1
        return np.array(inv_log_image, dtype=np.uint8)
