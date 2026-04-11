import cv2
import numpy as np

def apply_denoising(frame):
    """
    Applies light denoising using Gaussian Blur (Step 6).
    """
    return cv2.GaussianBlur(frame, (3, 3), 0)

def apply_contrast_normalization(frame):
    """
    Applies CLAHE for contrast normalization (Step 7) on BGR frames.
    """
    # For BGR, convert to LAB and apply CLAHE to L channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def normalize_frames(frames, target_size=(320, 240), denoise=True, normalize_contrast=True):
    """
    Complete normalization pipeline: denoising, contrast, resizing, and pixel scaling.
    target_size is (Width, Height).
    """
    processed_frames = []
    for frame in frames:
        # Step 5: Color Processing (Always Keep RGB/BGR)
        processed = frame.copy()
        
        # Step 6: Light Denoising
        if denoise:
            processed = apply_denoising(processed)
        
        # Step 7: Contrast Normalization
        if normalize_contrast:
            processed = apply_contrast_normalization(processed)
        
        # Step 3: Resize Frames (Rectangular 320x240)
        resized = cv2.resize(processed, target_size)
        
        # Step 4: Normalize Pixel Values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        processed_frames.append(normalized)
    
    return np.array(processed_frames)
