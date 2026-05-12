import tensorflow as tf
import numpy as np

def denoise_frame(image):
    """
    Applies manual 3x3 Gaussian blur denoising using TensorFlow depthwise convolution.
    The original NumPy shifting logic is commented out.
    """
    # # 3x3 Gaussian kernel
    # kernel = np.array([[1, 2, 1],
    #                    [2, 4, 2],
    #                    [1, 2, 1]], dtype=np.float32) / 16.0
    # 
    # h, w, c = image.shape
    # # Pad edges to keep output size identical to input (mode='edge' avoids dark borders)
    # padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge').astype(np.float32)
    # output = np.zeros_like(image, dtype=np.float32)
    # 
    # # Vectorized convolution loop over color channels
    # for c_idx in range(c):
    #     channel = padded[:, :, c_idx]
    #     out_channel = np.zeros((h, w), dtype=np.float32)
    #     # Shift and accumulate for the 3x3 window (vectorized summation)
    #     for i in range(3):
    #         for j in range(3):
    #             out_channel += channel[i:i+h, j:j+w] * kernel[i, j]
    #     output[:, :, c_idx] = out_channel
    #     
    # return output.astype(np.uint8)

    # 3x3 Gaussian kernel for TF
    # Shape for depthwise_conv2d: [filter_height, filter_width, in_channels, channel_multiplier]
    kernel_np = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]], dtype=np.float64) / 16.0
    
    # Expand kernel for all channels
    in_channels = image.shape[-1]
    kernel_tf = tf.constant(kernel_np[:, :, np.newaxis, np.newaxis], dtype=tf.float64)
    kernel_tf = tf.tile(kernel_tf, [1, 1, in_channels, 1])

    # Convert image to tensor and add batch dimension
    img_tf = tf.cast(image, dtype=tf.float64)
    img_tf = tf.expand_dims(img_tf, axis=0) # [1, H, W, C]

    # Pre-pad using SYMMETRIC (identical to 'edge' for 1px pad) to match NumPy exactly
    img_padded = tf.pad(img_tf, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')

    # Apply convolution (VALID because we already padded)
    output_tf = tf.nn.depthwise_conv2d(img_padded, kernel_tf, strides=[1, 1, 1, 1], padding='VALID')
    
    # Remove batch dimension and cast back to uint8
    output_tf = tf.squeeze(output_tf, axis=0)
    return tf.cast(output_tf, tf.uint8)
