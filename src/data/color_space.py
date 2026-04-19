import cv2

def optimize_color_space(frame, to_grayscale=False):
    """
    Step 7: Convert to Grayscale for structural analysis or maintain RGB.
    """
    if to_grayscale:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame
