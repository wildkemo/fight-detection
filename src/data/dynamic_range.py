import numpy as np
import cv2

def dynamic_range(frame):
    """
    Optimized Contrast Stretching.
    Uses OpenCV's NORM_MINMAX which is highly optimized.
    """
    # 0 to 255 linear mapping
    return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
