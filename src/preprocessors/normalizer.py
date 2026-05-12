import tensorflow as tf
import numpy as np

def normalize_frame(image):
    """
    Normalizes the frame. Scales to [0, 1] using TensorFlow.
    """
    # # Manual scalar multiplication using native numpy operations
    # return np.array(image, dtype=np.float32) * (1.0 / 255.0)
    
    img_tf = tf.cast(image, dtype=tf.float64)
    return img_tf * (1.0 / 255.0)

def denormalize_to_uint8(image):
    """
    Scales [0, 1] image back to [0, 255] uint8 for saving.
    """
    # # Manual unscaling and clipping
    # img = np.array(image) * 255.0
    # img[img < 0] = 0
    # img[img > 255] = 255
    # return img.astype(np.uint8)
    
    img_tf = image * 255.0
    img_clipped = tf.clip_by_value(img_tf, 0.0, 255.0)
    return tf.cast(img_clipped, tf.uint8)
