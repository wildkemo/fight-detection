import numpy as np

def build_sequences(frames, sequence_length=16):
    """
    Groups frames into fixed-length temporal sequences.
    """
    sequences = []
    num_frames = len(frames)
    
    # Simple chunking approach
    for i in range(0, num_frames, sequence_length):
        sequence = frames[i : i + sequence_length]
        
        # Only include complete sequences
        if len(sequence) == sequence_length:
            sequences.append(sequence)
            
    return np.array(sequences)
