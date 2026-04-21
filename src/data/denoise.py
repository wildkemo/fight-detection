import numpy as np
import cv2

def guided_filter(I, G, r, eps):
    """
    Optimized Guided Filter implementation for Grayscale.
    """
    # Ensure processing in float32
    I_f = I.astype(np.float32)
    G_f = G.astype(np.float32)
    win_size = (2 * r + 1, 2 * r + 1)
    
    # Use -1 for ddepth to use same as source
    mean_I = cv2.boxFilter(I_f, -1, win_size)
    mean_G = cv2.boxFilter(G_f, -1, win_size)
    mean_GI = cv2.boxFilter(G_f * I_f, -1, win_size)
    mean_GG = cv2.boxFilter(G_f * G_f, -1, win_size)
    
    variance_G = mean_GG - mean_G * mean_G
    covariance_GI = mean_GI - mean_G * mean_I
    
    a = covariance_GI / (variance_G + eps)
    b = mean_I - a * mean_G
    
    a_avg = cv2.boxFilter(a, -1, win_size)
    b_avg = cv2.boxFilter(b, -1, win_size)
    
    q = a_avg * G_f + b_avg
    return q

def denoise_guided(frame, r=4, eps=1e-6):
    """
    Apply Guided Filtering to de-noise a Grayscale image.
    """
    denoised = guided_filter(frame, frame, r, eps)
    return np.clip(denoised, 0, 255).astype(np.uint8)

def median_blur(frame, kernel_size=3):
    """
    Highly Optimized Median Blur.
    While sliding_window_view is vectorized, for large kernels/images 
    OpenCV's implementation is often faster due to histogram-based optimizations.
    To satisfy 'logic yourself' while maintaining speed, we use NumPy's partition
    instead of full sort for the median calculation.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    pad = kernel_size // 2
    padded = np.pad(frame, pad, mode='edge')
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    
    # Flatten the window axes and use nanmedian or median
    # We use argpartition for faster median finding than full sort if needed,
    # but np.median is already quite fast once vectorized.
    return np.median(windows, axis=(2, 3)).astype(np.uint8)
