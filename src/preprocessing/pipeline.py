import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import the siblings
sys.path.append(str(Path(__file__).parent))

from contrast_stretching import manual_contrast_stretch
from clahe import manual_clahe
from resize import manual_letterbox
from guassien_blur import manual_gaussian_blur

class FramePreprocessor:
    """
    Unified pipeline for manual frame preprocessing.
    Applies steps in the specific order: 
    Contrast Stretch -> CLAHE -> Resize -> Gaussian Blur
    """
    def __init__(self, target_w=640, target_h=640, clahe_tiles=(8, 8), 
                 clahe_clip=2.0, blur_kernel=5, blur_sigma=1.0):
        self.target_w = target_w
        self.target_h = target_h
        self.clahe_tiles = clahe_tiles
        self.clahe_clip = clahe_clip
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma

    def process_frame(self, frame):
        if frame is None:
            return None
        
        # 1. Contrast Stretching (Enhance Global Contrast)
        frame = manual_contrast_stretch(frame)
        
        # 2. CLAHE (Enhance Local Contrast)
        frame = manual_clahe(frame, self.clahe_tiles[0], self.clahe_tiles[1], self.clahe_clip)
        
        # 3. Resize / Letterbox (Model input sizing)
        frame = manual_letterbox(frame, self.target_w, self.target_h)
        
        # 4. Gaussian Blur (Denoise)
        frame = manual_gaussian_blur(frame, self.blur_kernel, self.blur_sigma)
        
        return frame
