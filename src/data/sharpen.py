import cv2
import numpy as np

def sharpen(frame):
    """
    Optimized Unsharp Masking.
    Formula: Result = Original + Amount * (Original - Blurred)
    Simplified to: Result = (1 + Amount) * Original - Amount * Blurred
    """
    # 1. Faster blurring with smaller kernel or pre-calculated sigma
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # 2. Vectorized arithmetic is faster than multiple calls to addWeighted/subtract
    # We use float32 for calculation then clip and convert back once
    amount = 5.0
    # Equivalent to: frame + amount * (frame - blurred)
    enhanced = (1.0 + amount) * frame.astype(np.float32) - amount * blurred.astype(np.float32)
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)
