import cv2

def resize(frame, width=224, height=224):
    """
    Step 2: Resize all frames to a uniform dimension.
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
