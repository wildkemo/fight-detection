import numpy as np

def normalize_frame(image):
    """
    Normalizes the frame. Currently scales to [0, 1].
    
    Args:
        image (numpy.ndarray): The input image.
        
    Returns:
        numpy.ndarray: The normalized image.
    """
    # Scale to [0, 1]
    # Note: If saving as JPEG, we might skip this or scale back to [0, 255]
    return image.astype(np.float32) / 255.0

def denormalize_to_uint8(image):
    """
    Scales [0, 1] image back to [0, 255] uint8 for saving.
    """
    return (image * 255.0).astype(np.uint8)
