import numpy as np
import cv2

def guided_filter(I, G, r, eps):
    """
    High-performance Guided Filter implementation for Grayscale.
    """
    I = I.astype(np.float32)
    G = G.astype(np.float32)
    win_size = (2 * r + 1, 2 * r + 1)
    
    mean_I = cv2.boxFilter(I, -1, win_size)
    mean_G = cv2.boxFilter(G, -1, win_size)
    mean_GI = cv2.boxFilter(G * I, -1, win_size)
    mean_GG = cv2.boxFilter(G * G, -1, win_size)
    
    variance_G = mean_GG - mean_G * mean_G
    covariance_GI = mean_GI - mean_G * mean_I
    
    a = covariance_GI / (variance_G + eps)
    b = mean_I - a * mean_G
    
    a_avg = cv2.boxFilter(a, -1, win_size)
    b_avg = cv2.boxFilter(b, -1, win_size)
    
    q = a_avg * G + b_avg
    return q

def denoise_guided(frame, r=4, eps=1e-6):
    """
    Apply Guided Filtering to de-noise a Grayscale image.
    """
    denoised = guided_filter(frame, frame, r, eps)
    return np.clip(denoised, 0, 255).astype(np.uint8)

def median_blur(frame, kernel_size=3):
    """
    Optimized Vectorized Median Blur implementation.
    Uses sliding window view for high performance without per-pixel loops.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    # 1. Padding to maintain dimensions
    pad = kernel_size // 2
    padded_frame = np.pad(frame, pad, mode='edge')
    
    # 2. Extract all windows at once (Vectorized)
    # Resulting shape: (img_h, img_w, k_h, k_w)
    windows = sliding_window_view(padded_frame, (kernel_size, kernel_size))
    
    # 3. Compute median along the window axes (2 and 3)
    # This replaces the nested Python loops with optimized NumPy C-code
    return np.median(windows, axis=(2, 3)).astype(np.uint8)
