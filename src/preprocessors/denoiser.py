import numpy as np

def denoise_frame(image):
    """
    Applies manual 3x3 Gaussian blur denoising without using ready-made functions.
    Implemented using vectorized 2D convolution with array shifting.
    """
    # 3x3 Gaussian kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16.0
    
    h, w, c = image.shape
    # Pad edges to keep output size identical to input (mode='edge' avoids dark borders)
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge').astype(np.float32)
    output = np.zeros_like(image, dtype=np.float32)
    
    # Vectorized convolution loop over color channels
    for c_idx in range(c):
        channel = padded[:, :, c_idx]
        out_channel = np.zeros((h, w), dtype=np.float32)
        # Shift and accumulate for the 3x3 window (vectorized summation)
        for i in range(3):
            for j in range(3):
                out_channel += channel[i:i+h, j:j+w] * kernel[i, j]
        output[:, :, c_idx] = out_channel
        
    return output.astype(np.uint8)
