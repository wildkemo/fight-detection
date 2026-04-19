import cv2
import numpy as np

def apply_edge_enhancement(frame):
    """
    Step 6: Aggressive High-Pass Sharpening.
    
    Applies a convolutional kernel to dramatically boost edges,
    creating a very sharp, almost threshold-like appearance.
    """
    # Strong sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
                       
    return cv2.filter2D(frame, -1, kernel)
