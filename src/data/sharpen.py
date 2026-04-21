import cv2
import numpy as np

def sharpen(frame):
    """
    Step 6: Unsharp Masking (Proportional Sharpening).
    
    Extracts high-frequency details by subtracting a blurred version 
    of the image from the original, then adds them back.
    
    Formula: Result = Original + Amount * (Original - Blurred)
    """
    # 1. Create a blurred version of the image (Low-pass filter)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # 2. Extract the details (High-pass component)
    details = cv2.subtract(frame, blurred)
    
    # 3. Add details back to the original image
    # Increased amount for stronger sharpening (aggressive look)
    amount = 5.0
    enhanced_frame = cv2.addWeighted(frame, 1.0, details, amount, 0)
    
    return enhanced_frame
