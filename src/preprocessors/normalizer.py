import numpy as np

def normalize_frame(image):
    """
    Normalizes the frame. Scales to [0, 1] without library functions.
    """
    # Manual scalar multiplication using native numpy operations
    return np.array(image, dtype=np.float32) * (1.0 / 255.0)

def denormalize_to_uint8(image):
    """
    Scales [0, 1] image back to [0, 255] uint8 for saving.
    """
    # Manual unscaling and clipping
    img = np.array(image) * 255.0
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)
