import os
import cv2

def storage(frame, video_path, output_root, saved_count, burst_size):
    """
    Step 9: Save processed frames into a structured directory hierarchy.
    output/<category>/<video_name>/burst_<index>/frame_<index>.jpg
    """
    # Extract category and video name for the folder structure
    category = os.path.basename(os.path.dirname(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_root, category, video_name)
    
    # Calculate burst index
    burst_index = saved_count // burst_size
    burst_dir = os.path.join(video_output_dir, f"burst_{burst_index:03d}")
    
    # Ensure directory exists
    if not os.path.exists(burst_dir):
        os.makedirs(burst_dir, exist_ok=True)
    
    # Save the frame
    frame_filename = f"frame_{saved_count:04d}.jpg"
    out_path = os.path.join(burst_dir, frame_filename)
    cv2.imwrite(out_path, frame)
    return out_path
