import cv2

def color_space(frame):
    """
    Step 7 (Now Step 1.5): Convert to Grayscale.
    Expects a BGR image and returns a Grayscale image.
    """
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame
