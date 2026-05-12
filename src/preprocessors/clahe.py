import tensorflow as tf
import numpy as np

def bgr_to_ycbcr(image):
    """
    Manual BGR to YCbCr conversion using TensorFlow.
    """
    # # Transformation matrix (standard BT.601)
    # # Using float64 for intermediate precision
    # img = image.astype(np.float64)
    # b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    # 
    # y = 0.299 * r + 0.587 * g + 0.114 * b
    # cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    # cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    # 
    # return np.stack([y, cb, cr], axis=-1)
    
    img = tf.cast(image, tf.float64)
    b, g, r = tf.unstack(img, axis=-1)
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    
    return tf.stack([y, cb, cr], axis=-1)

def ycbcr_to_bgr(ycbcr):
    """
    Manual YCbCr to BGR conversion using TensorFlow.
    """
    # img = ycbcr.astype(np.float64)
    # y, cb, cr = img[:,:,0], img[:,:,1], img[:,:,2]
    # 
    # r = y + 1.402 * (cr - 128)
    # g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    # b = y + 1.772 * (cb - 128)
    # 
    # bgr = np.stack([b, g, r], axis=-1)
    # return np.clip(bgr, 0, 255).astype(np.uint8)
    
    img = tf.cast(ycbcr, tf.float64)
    y, cb, cr = tf.unstack(img, axis=-1)
    
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    
    bgr = tf.stack([b, g, r], axis=-1)
    bgr = tf.clip_by_value(bgr, 0.0, 255.0)
    return tf.cast(bgr, tf.uint8)

def get_clahe_mapping(tile, clip_limit):
    """
    Calculates the contrast-limited histogram and CDF mapping for a single tile.
    """
    # 1. Compute Histogram
    hist, _ = np.histogram(tile, bins=256, range=(0, 256))
    
    # 2. Clip Histogram
    excess = np.sum(np.maximum(hist - clip_limit, 0))
    hist = np.minimum(hist, clip_limit)
    
    # 3. Redistribute Excess
    redistribution = excess / 256.0
    hist = hist + redistribution
    
    # 4. Compute Normalized CDF
    cdf = np.cumsum(hist)
    cdf_min = cdf.min()
    cdf_max = cdf.max()
    
    if cdf_max - cdf_min == 0:
        return np.arange(256)
        
    # Scale CDF to [0, 255]
    return (cdf - cdf_min) * 255.0 / (cdf_max - cdf_min)

def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    Full manual implementation of CLAHE.
    """
    # Step 1: Convert to YCbCr to process Luminance (Y) channel
    ycbcr = bgr_to_ycbcr(image)
    y_tf = ycbcr[:, :, 0]
    y_np = tf.cast(y_tf, tf.uint8).numpy()
    h, w = y_np.shape
    
    gh, gw = grid_size
    tile_h = int(np.ceil(h / gh))
    tile_w = int(np.ceil(w / gw))
    
    # Pad Y channel to be perfectly divisible by grid size if needed
    pad_h = gh * tile_h - h
    pad_w = gw * tile_w - w
    y_padded = np.pad(y_np, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    # Step 2: Compute mappings for each tile (Keep in NumPy for histogram complexity)
    mappings = np.zeros((gh, gw, 256), dtype=np.float64)
    actual_clip_limit = max(1.0, clip_limit * (tile_h * tile_w) / 256.0)
    
    for i in range(gh):
        for j in range(gw):
            tile = y_padded[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            mappings[i, j] = get_clahe_mapping(tile, actual_clip_limit)
            
    # Step 3: Bilinear Interpolation (Optimized via TensorFlow)
    # # We interpolate between tile centers
    # ty = (np.arange(h) + 0.5) / tile_h - 0.5
    # tx = (np.arange(w) + 0.5) / tile_w - 0.5
    # ty = np.clip(ty, 0, gh - 1)
    # tx = np.clip(tx, 0, gw - 1)
    # y0 = np.floor(ty).astype(int)
    # y1 = np.minimum(y0 + 1, gh - 1)
    # x0 = np.floor(tx).astype(int)
    # x1 = np.minimum(x0 + 1, gw - 1)
    # dy = (ty - y0)[:, None]
    # dx = (tx - x0)[None, :]
    # Y0, Y1 = y0[:, None], y1[:, None]
    # X0, X1 = x0[None, :], x1[None, :]
    # map00 = mappings[Y0, X0, y]
    # map01 = mappings[Y0, X1, y]
    # map10 = mappings[Y1, X0, y]
    # map11 = mappings[Y1, X1, y]
    # out_y = (map00 * (1 - dy) * (1 - dx) +
    #          map01 * (1 - dy) * dx +
    #          map10 * dy * (1 - dx) +
    #          map11 * dy * dx)

    # Convert mappings to TF tensor for GPU lookup
    mappings_tf = tf.convert_to_tensor(mappings, dtype=tf.float64)
    
    # Calculate coordinates
    ty = (tf.range(h, dtype=tf.float64) + 0.5) / float(tile_h) - 0.5
    tx = (tf.range(w, dtype=tf.float64) + 0.5) / float(tile_w) - 0.5
    ty = tf.clip_by_value(ty, 0.0, float(gh - 1))
    tx = tf.clip_by_value(tx, 0.0, float(gw - 1))
    
    y0 = tf.cast(tf.floor(ty), tf.int32)
    y1 = tf.minimum(y0 + 1, gh - 1)
    x0 = tf.cast(tf.floor(tx), tf.int32)
    x1 = tf.minimum(x0 + 1, gw - 1)
    
    dy = tf.expand_dims(ty - tf.cast(y0, tf.float64), axis=1)
    dx = tf.expand_dims(tx - tf.cast(x0, tf.float64), axis=0)
    
    # Create 2D grids of indices
    Y0, X0 = tf.meshgrid(y0, x0, indexing='ij')
    Y1, X1 = tf.meshgrid(y1, x1, indexing='ij')
    
    # Advanced Indexing (equivalent to mappings[Y, X, y])
    # mappings_tf is [gh, gw, 256], y_tf is [h, w]
    def gather_mappings(Y, X, val):
        indices = tf.stack([Y, X, tf.cast(val, tf.int32)], axis=-1)
        return tf.gather_nd(mappings_tf, indices)

    map00 = gather_mappings(Y0, X0, y_tf)
    map01 = gather_mappings(Y0, X1, y_tf)
    map10 = gather_mappings(Y1, X0, y_tf)
    map11 = gather_mappings(Y1, X1, y_tf)

    out_y = (map00 * (1.0 - dy) * (1.0 - dx) +
             map01 * (1.0 - dy) * dx +
             map10 * dy * (1.0 - dx) +
             map11 * dy * dx)
             
    # Step 4: Reconstruct BGR
    # Stack back to YCbCr tensor
    cb_cr = ycbcr[:, :, 1:]
    ycbcr_new = tf.concat([tf.expand_dims(out_y, axis=-1), cb_cr], axis=-1)
    return ycbcr_to_bgr(ycbcr_new)
