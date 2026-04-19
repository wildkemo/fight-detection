import cv2
import numpy as np

def apply_contrast_enhancement(frame):
    """
    Step 5: Multi-Scale Adaptive Histogram Equalization.
    
    Combines two CLAHE passes at different scales (fine and coarse) 
    to enhance both local texture and global structure.
    """
    # 1. Convert to LAB color space to process the Luminance (L) channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 2. Fine-scale CLAHE (captures small textures/edges)
    clahe_fine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_fine = clahe_fine.apply(l)
    
    # 3. Coarse-scale CLAHE (captures larger structural shapes/shadows)
    clahe_coarse = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    l_coarse = clahe_coarse.apply(l)
    
    # 4. Blend the scales (50/50 mix)
    l_final = cv2.addWeighted(l_fine, 0.5, l_coarse, 0.5, 0)
    
    # 5. Merge back and convert to BGR
    limg = cv2.merge((l_final, a, b))
    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_frame
