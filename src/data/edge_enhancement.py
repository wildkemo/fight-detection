import cv2
import numpy as np

def apply_edge_enhancement(frame, radius=8, eps=0.01, enhancement_factor=1.5):
    """
    Step 6: Apply Guided Filter Detail Enhancement.
    
    This technique enhances details/edges without the halo artifacts of Unsharp Masking.
    It works by:
    1. Applying an edge-preserving Guided Filter (base layer).
    2. Subtracting base from original to get the detail layer.
    3. Adding the boosted detail layer back to the original.
    """
    # Ensure frame is float32 for processing
    img = frame.astype(np.float32) / 255.0
    
    # Simple Guided Filter Implementation
    def guided_filter(I, p, r, eps):
        mean_I = cv2.boxFilter(I, -1, (r, r))
        mean_p = cv2.boxFilter(p, -1, (r, r))
        mean_Ip = cv2.boxFilter(I * p, -1, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I * I, -1, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, (r, r))
        mean_b = cv2.boxFilter(b, -1, (r, r))

        return mean_a * I + mean_b

    # If RGB, process each channel or use grayscale as guide
    if len(img.shape) == 3:
        # Use a grayscale version as the guide I
        guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        base = np.zeros_like(img)
        for i in range(3):
            base[:,:,i] = guided_filter(guide, img[:,:,i], radius, eps)
    else:
        base = guided_filter(img, img, radius, eps)
    
    # Detail Layer = Original - Base
    detail = img - base
    
    # Enhanced Image = Original + (Enhancement Factor * Detail)
    enhanced = img + (enhancement_factor * detail)
    
    # Clip and return as uint8
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    return enhanced
