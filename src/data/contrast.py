import cv2
import numpy as np

def manual_histogram_equalization(img):
    """
    Manual Global Histogram Equalization for Grayscale images.
    """
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    
    num = (cdf_m - cdf_m.min()) * 255
    den = cdf_m.max() - cdf_m.min()
    
    if den == 0:
        return img
        
    cdf_m = num / den
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf_final[img]

def contrast(frame):
    """
    Step 5: Global Histogram Equalization.
    Expects a Grayscale image.
    """
    return manual_histogram_equalization(frame)
