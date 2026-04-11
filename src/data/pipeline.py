import os
import cv2
import numpy as np
from .extract_frames import extract_frames
from .normalize import normalize_frames
from .sequence_builder import build_sequences

def process_video(video_path, output_base_dir, label, target_fps=5, sequence_length=16, target_size=(320, 240)):
    """
    Runs the full preprocessing pipeline on a single video file.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create label-specific output directory
    output_dir = os.path.join(output_base_dir, "processed_data", label)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"--- Processing Video: {video_name} (Label: {label}) ---")
    
    # 1. Step 2: Extract Frames
    frames = extract_frames(video_path, target_fps=target_fps)
    print(f"Extracted {len(frames)} frames at {target_fps} FPS.")
    
    # 2. Step 3-7: Full Normalization (Denoise, Contrast, Resize, Pixel Scaling)
    normalized_frames = normalize_frames(
        frames, 
        target_size=target_size, 
        denoise=True, 
        normalize_contrast=True
    )
    print(f"Normalized frames shape: {normalized_frames.shape}")
    
    # 3. Step 8: Sequence Construction
    sequences = build_sequences(normalized_frames, sequence_length=sequence_length)
    print(f"Created {len(sequences)} sequences of length {sequence_length}.")
    
    # 4. Step 9 & 10: Label Assignment & Save Processed Data
    npy_output_path = os.path.join(output_dir, f"{video_name}.npy")
    np.save(npy_output_path, sequences)
    print(f"Saved sequences to: {npy_output_path}")
    
    # 5. Step 11: Sanity Check (Save all sequences as PNGs)
    sanity_check_base = os.path.join(output_base_dir, "sanity_check", label, video_name)
    if not os.path.exists(sanity_check_base):
        os.makedirs(sanity_check_base)
    
    for s_idx, seq in enumerate(sequences):
        seq_dir = os.path.join(sanity_check_base, f"sequence_{s_idx}")
        if not os.path.exists(seq_dir):
            os.makedirs(seq_dir)
        
        for f_idx, frame in enumerate(seq):
            # Convert [0, 1] to [0, 255] for image saving
            img = (frame * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(seq_dir, f"frame_{f_idx:03d}.png"), img)
    
    if len(sequences) > 0:
        print(f"Saved sanity check sequences to: {sanity_check_base}")

    return sequences

if __name__ == "__main__":
    # Standard test run on V_19.mp4
    video_path = "dataset/Real Life Violence Dataset/Violence/V_19.mp4"
    output_base_dir = "tmp"
    label = "violence"
    
    if os.path.exists(video_path):
        process_video(video_path, output_base_dir, label)
    else:
        print(f"Error: Video file not found at {video_path}")
