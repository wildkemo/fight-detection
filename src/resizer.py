import cv2
import numpy as np

def resize_with_padding(image, target_size=(224, 224)):
    """
    Resizes an image to the target size while maintaining aspect ratio using padding.
    
    Args:
        image (numpy.ndarray): The input image.
        target_size (tuple): The desired (width, height).
        
    Returns:
        numpy.ndarray: The resized and padded image.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate aspect ratios
    aspect_ratio = w / h
    target_aspect_ratio = target_w / target_h
    
    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target: resize by width
        new_w = target_w
        new_h = int(new_w / aspect_ratio)
    else:
        # Image is taller than target: resize by height
        new_h = target_h
        new_w = int(new_h * aspect_ratio)
        
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a black canvas of target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate offsets to center the image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas
