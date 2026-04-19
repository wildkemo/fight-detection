import cv2
import numpy as np

def sanity_check(frame):
    """
    Step 8: Visually inspect processed frames (metadata check).
    """
    if frame is None or frame.size == 0:
        return False
    # Ensure the frame has sufficient variance (not just a solid block)
    if np.std(frame) < 2:
        return False
    return True
