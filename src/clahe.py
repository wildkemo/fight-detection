import cv2

def apply_clahe(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the frame.
    
    Args:
        image (numpy.ndarray): The input image (BGR).
        
    Returns:
        numpy.ndarray: The contrast-enhanced image.
    """
    # CLAHE is usually applied to the L channel of LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final
