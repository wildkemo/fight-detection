import cv2

def denoise_frame(frame):
    """
    Step 3: Apply spatial filters to reduce sensor noise.
    """
    # Using Median Blur to preserve edges while removing salt-and-pepper noise
    return cv2.medianBlur(frame, 5)
