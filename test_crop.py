import numpy as np

def crop_frame(fh, fw, pw, ph):
    frame = np.zeros((fh, fw, 3), dtype=np.uint8)
    target_ratio = pw / ph
    current_ratio = fw / fh
    
    if current_ratio < target_ratio - 0.01:
        new_h = int(fw / target_ratio)
        crop_top = (fh - new_h) // 2
        frame = frame[crop_top:crop_top+new_h, :]
    elif current_ratio > target_ratio + 0.01:
        new_w = int(fh * target_ratio)
        crop_left = (fw - new_w) // 2
        frame = frame[:, crop_left:crop_left+new_w]
    return frame.shape

print("640x480 to 854x480:", crop_frame(480, 640, 854, 480))
print("1920x1080 to 854x480:", crop_frame(1080, 1920, 854, 480))
print("854x480 to 854x480:", crop_frame(480, 854, 854, 480))
