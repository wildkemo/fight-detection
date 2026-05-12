import numpy as np

def bgr_to_ycbcr(image):
    """
    Manual BGR to YCbCr conversion.
    """
    # Transformation matrix (standard BT.601)
    # Using float64 for intermediate precision
    img = image.astype(np.float64)
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    
    return np.stack([y, cb, cr], axis=-1)

def ycbcr_to_bgr(ycbcr):
    """
    Manual YCbCr to BGR conversion.
    """
    img = ycbcr.astype(np.float64)
    y, cb, cr = img[:,:,0], img[:,:,1], img[:,:,2]
    
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    
    bgr = np.stack([b, g, r], axis=-1)
    return np.clip(bgr, 0, 255).astype(np.uint8)

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
    y = ycbcr[:, :, 0].astype(np.uint8)
    h, w = y.shape
    
    gh, gw = grid_size
    tile_h = int(np.ceil(h / gh))
    tile_w = int(np.ceil(w / gw))
    
    # Pad Y channel to be perfectly divisible by grid size if needed
    pad_h = gh * tile_h - h
    pad_w = gw * tile_w - w
    y_padded = np.pad(y, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    # Step 2: Compute mappings for each tile
    mappings = np.zeros((gh, gw, 256))
    # Scale clip limit according to tile size
    # clip_limit of 2.0 means 2x the average bin height
    actual_clip_limit = max(1.0, clip_limit * (tile_h * tile_w) / 256.0)
    
    for i in range(gh):
        for j in range(gw):
            tile = y_padded[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            mappings[i, j] = get_clahe_mapping(tile, actual_clip_limit)
            
    # Step 3: Bilinear Interpolation to smooth boundaries
    # Create coordinate grids
    # We interpolate between tile centers
    ty = (np.arange(h) + 0.5) / tile_h - 0.5
    tx = (np.arange(w) + 0.5) / tile_w - 0.5
    
    # Clip to mapping grid bounds
    ty = np.clip(ty, 0, gh - 1)
    tx = np.clip(tx, 0, gw - 1)
    
    y0 = np.floor(ty).astype(int)
    y1 = np.minimum(y0 + 1, gh - 1)
    x0 = np.floor(tx).astype(int)
    x1 = np.minimum(x0 + 1, gw - 1)
    
    # Interpolation weights
    dy = (ty - y0)[:, None]
    dx = (tx - x0)[None, :]
    
    # Optimized lookup using advanced indexing
    Y0, Y1 = y0[:, None], y1[:, None]
    X0, X1 = x0[None, :], x1[None, :]
    
    # Apply mappings
    map00 = mappings[Y0, X0, y]
    map01 = mappings[Y0, X1, y]
    map10 = mappings[Y1, X0, y]
    map11 = mappings[Y1, X1, y]
    
    # Combine using bilinear interpolation
    out_y = (map00 * (1 - dy) * (1 - dx) +
             map01 * (1 - dy) * dx +
             map10 * dy * (1 - dx) +
             map11 * dy * dx)
             
    # Step 4: Reconstruct BGR
    ycbcr[:, :, 0] = out_y
    return ycbcr_to_bgr(ycbcr)
