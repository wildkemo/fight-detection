import numpy as np

def manual_resize(image, target_size):
    """
    Manual image resizing implementation using Bilinear Interpolation.
    
    Args:
        image (numpy.ndarray): Input image (H, W, C).
        target_size (tuple): Desired (width, height).
        
    Returns:
        numpy.ndarray: Resized image.
    """
    new_w, new_h = target_size
    h, w, c = image.shape
    
    # Calculate scale factors
    x_scale = w / new_w
    y_scale = h / new_h
    
    y_indices = np.arange(new_h)
    x_indices = np.arange(new_w)
    
    # Calculate geometric mapping to original image (0.5 offset for center-alignment)
    y_coords = np.clip((y_indices + 0.5) * y_scale - 0.5, 0, h - 1)
    x_coords = np.clip((x_indices + 0.5) * x_scale - 0.5, 0, w - 1)
    
    y0 = np.floor(y_coords).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(x_coords).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    
    # Calculate interpolation weights
    dy = (y_coords - y0).reshape(-1, 1, 1)
    dx = (x_coords - x0).reshape(1, -1, 1)
    
    # Advanced vectorized indexing to get the 4 corner pixels
    Y0 = y0[:, None]
    Y1 = y1[:, None]
    X0 = x0[None, :]
    X1 = x1[None, :]
    
    Ia = image[Y0, X0]
    Ib = image[Y0, X1]
    Ic = image[Y1, X0]
    Id = image[Y1, X1]
    
    # Bilinear interpolation formula
    wa = (1 - dx) * (1 - dy)
    wb = dx * (1 - dy)
    wc = (1 - dx) * dy
    wd = dx * dy
    
    resized = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return resized.astype(np.uint8)

def resize_with_padding(image, target_size=(224, 224)):
    """
    Resizes manually while maintaining aspect ratio using padding.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    aspect_ratio = w / h
    target_aspect_ratio = target_w / target_h
    
    if aspect_ratio > target_aspect_ratio:
        new_w = target_w
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(new_h * aspect_ratio)
        
    # Use custom resize logic instead of cv2.resize
    resized = manual_resize(image, (new_w, new_h))
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas
