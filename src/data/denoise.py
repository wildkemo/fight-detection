import cv2

def denoise_frame(frame):
    """
    Step 3: Apply spatial filters to reduce sensor noise.
    Using Bilateral Filter to strongly preserve edges while removing noise,
    preventing the blurring caused by Median Blur.
    """
    return cv2.bilateralFilter(frame, 5, 50, 50)
