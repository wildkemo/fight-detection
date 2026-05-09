import cv2

def denoise_frame(image):
    """
    Applies mild denoising to the frame.
    
    Args:
        image (numpy.ndarray): The input image.
        
    Returns:
        numpy.ndarray: The denoised image.
    """
    # Using Gaussian Blur for mild denoising. 
    # Alternative: cv2.fastNlMeansDenoisingColored for better but slower denoising.
    # Given the CPU optimization goal, a small Gaussian kernel is efficient.
    return cv2.GaussianBlur(image, (3, 3), 0)
