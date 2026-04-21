import cv2
import numpy as np

def manual_histogram_equalization(img):
    """
    Optimized Global Histogram Equalization using LUT.
    """
    # 1. Calculate the histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    # 2. Calculate the Normalized CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    
    # 3. Create a Look-Up Table (LUT)
    lut = np.round(cdf_normalized).astype('uint8')
    
    # 4. Apply mapping using highly optimized OpenCV primitive
    return cv2.LUT(img, lut)

def contrast(frame):
    """
    Step 5: Global Histogram Equalization.
    Expects a Grayscale image.
    """
    return manual_histogram_equalization(frame)
