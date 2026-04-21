import numpy as np


def median_filter_gray(image, kernel_size=3):
    pad = kernel_size // 2
    h, w = image.shape

    # Pad image
    padded = np.pad(image, pad, mode='edge')

    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.median(window)

    return output


def denoise_frame(frame, kernel_size=3):
    """
    Apply manual median filtering.
    Works for both grayscale and color images.
    """

    # Grayscale
    if len(frame.shape) == 2:
        return median_filter_gray(frame, kernel_size)

    # Color image → apply per channel
    output = np.zeros_like(frame)

    for c in range(frame.shape[2]):
        output[:, :, c] = median_filter_gray(frame[:, :, c], kernel_size)

    return output