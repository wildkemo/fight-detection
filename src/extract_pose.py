import os
import cv2
import json
import argparse
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

class RTMPoseInferencer:
    """
    Production-ready RTMPose ONNX inference with aspect-ratio preserving 
    preprocessing and vectorized SimCC decoding.
    """
    def __init__(self, model_path, device='cpu', num_threads=4):
        providers = ['CPUExecutionProvider']
        if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider']
        
        # Performance tuning for CPU
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # Get model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # input_shape is typically [1, 3, H, W]
        self.model_h, self.model_w = self.input_shape[2], self.input_shape[3]
        
        # Normalization constants (ImageNet)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)

    def preprocess(self, img, bbox):
        """
        Extracts a person crop with aspect-ratio preservation using affine transform.
        Returns normalized crop, the transform matrix, and its inverse.
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        center = np.array([x1 + w * 0.5, y1 + h * 0.5], dtype=np.float32)
        aspect_ratio = self.model_w / self.model_h
        
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        
        scale = np.array([w, h], dtype=np.float32) * 1.25
        trans = self.get_affine_transform(center, scale, 0, (self.model_w, self.model_h))
        
        # Warp image
        crop = cv2.warpAffine(img, trans, (self.model_w, self.model_h), flags=cv2.INTER_LINEAR)
        
        # Color conversion & Normalization
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = (crop.astype(np.float32) - self.mean) / self.std
        crop = crop.transpose(2, 0, 1)  # HWC -> CHW
        
        # Precompute inverse for restoration performance
        inv_trans = cv2.invertAffineTransform(trans)
        
        return crop[None, ...], trans, inv_trans

    def get_affine_transform(self, center, scale, rot, output_size):
        """Standard top-down affine transform calculation."""
        src_w, src_h = scale
        dst_w, dst_h = output_size

        rot_rad = np.pi * rot / 180
        src_dir = self._rotate_point([0, src_h * -0.5], rot_rad)
        dst_dir = np.array([0, dst_h * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        return cv2.getAffineTransform(np.float32(src), np.float32(dst))

    def _rotate_point(self, pt, angle_rad):
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        return [pt[0] * cs - pt[1] * sn, pt[0] * sn + pt[1] * cs]

    def _get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def decode_simcc(self, outputs):
        """
        Vectorized SimCC decoding with automatic coordinate scaling.
        Handles models where SimCC output resolution is a multiple of input resolution.
        """
        # RTMPose models usually have simcc_x and simcc_y as separate outputs.
        # We identify them by matching their lengths to model width/height.
        o1, o2 = outputs[0], outputs[1]
        
        # Check which output matches which axis (robust to output order)
        if abs(o1.shape[2] / self.model_w - 2.0) < 0.1 or o1.shape[2] == self.model_w:
            simcc_x, simcc_y = o1, o2
        else:
            simcc_x, simcc_y = o2, o1

        # Vectorized argmax and max
        x_indices = np.argmax(simcc_x, axis=2)  # [batch, 17]
        y_indices = np.argmax(simcc_y, axis=2)  # [batch, 17]
        
        x_scores = np.max(simcc_x, axis=2)
        y_scores = np.max(simcc_y, axis=2)
        scores = (x_scores + y_scores) / 2.0
        
        # Scale indices to model pixels
        # (SimCC length is typically model_res * 2)
        x = x_indices.astype(np.float32) / (simcc_x.shape[2] / self.model_w)
        y = y_indices.astype(np.float32) / (simcc_y.shape[2] / self.model_h)
        
        return np.stack([x, y, scores], axis=2)

    def infer_batch(self, batch_crops):
        """Runs inference on a batch of normalized crops."""
        if not batch_crops:
            return []
        
        # Strictly maintain NCHW batching
        input_tensor = np.concatenate(batch_crops, axis=0)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Safety assertions
        assert len(outputs) >= 2, "Expected SimCC X and Y outputs"
        assert outputs[0].shape[0] == input_tensor.shape[0], "Batch size mismatch"
        
        # Vectorized decoding
        batch_results = self.decode_simcc(outputs)
        return [batch_results[i] for i in range(batch_results.shape[0])]

def restore_coords(kpts, inv_trans):
    """
    Maps local crop keypoints back to full frame coordinates 
    using a precomputed inverse affine transform matrix.
    """
    # kpts: [17, 3] (x, y, conf)
    coords = kpts[:, :2]
    ones = np.ones((len(coords), 1), dtype=np.float32)
    coords_hom = np.hstack([coords, ones])
    
    # Apply inverse transform
    coords_restored = (inv_trans @ coords_hom.T).T
    
    kpts_restored = kpts.copy()
    kpts_restored[:, :2] = coords_restored
    return kpts_restored

def process_video_poses(video_path, track_data, inferencer, output_path, conf_thresh=0.3, alpha=0.7):
    """
    Main video processing loop with batching, robust coordinate restoration, 
    temporal smoothing, and normalization.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_map = {}
    for tid, frames in track_data['tracks'].items():
        for f in frames:
            f_idx = f['frame_idx']
            if f_idx not in frame_map:
                frame_map[f_idx] = []
            frame_map[f_idx].append({'tid': tid, 'bbox': f['bbox']})

    pose_results = {tid: [] for tid in track_data['tracks'].keys()}
    # Dictionary to store previous smoothed keypoints for each track
    prev_smoothed_kpts = {} 
    
    sorted_frames = sorted(frame_map.keys())
    
    if not sorted_frames:
        cap.release()
        return

    pbar = tqdm(total=len(sorted_frames), desc=f"Pose: {os.path.basename(video_path)}")
    
    curr_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if curr_idx in frame_map:
            targets = frame_map[curr_idx]
            batch_crops = []
            batch_meta = []
            
            for t in targets:
                bbox = t['bbox']
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                    
                crop, _, inv_trans = inferencer.preprocess(frame, bbox)
                batch_crops.append(crop)
                batch_meta.append({'tid': t['tid'], 'inv_trans': inv_trans})
            
            if batch_crops:
                results = inferencer.infer_batch(batch_crops)
                
                for kpts_local, meta in zip(results, batch_meta):
                    tid = meta['tid']
                    kpts_full = restore_coords(kpts_local, meta['inv_trans'])
                    
                    # 2. Temporal Smoothing & Low-Confidence Handling
                    if tid not in prev_smoothed_kpts:
                        prev_smoothed_kpts[tid] = kpts_full.copy()
                        smoothed_kpts = kpts_full
                    else:
                        prev_kpts = prev_smoothed_kpts[tid]
                        smoothed_kpts = kpts_full.copy()
                        
                        for j in range(len(kpts_full)):
                            x, y, conf = kpts_full[j, :3]
                            px, py, _ = prev_kpts[j, :3]
                            
                            if conf < conf_thresh:
                                # Joint unreliable: Hold previous position
                                smoothed_kpts[j, 0] = px
                                smoothed_kpts[j, 1] = py
                            else:
                                # Exponential smoothing
                                smoothed_kpts[j, 0] = alpha * x + (1 - alpha) * px
                                smoothed_kpts[j, 1] = alpha * y + (1 - alpha) * py
                        
                        prev_smoothed_kpts[tid] = smoothed_kpts.copy()
                    
                    # 3. Final Formatting [17, 3]
                    final_kpts = []
                    for x, y, conf in smoothed_kpts:
                        final_kpts.append([float(x), float(y), float(conf)])
                    
                    pose_results[tid].append({
                        "frame_idx": curr_idx,
                        "keypoints": final_kpts
                    })
            
            pbar.update(1)
        
        curr_idx += 1

    cap.release()
    pbar.close()

    with open(output_path, 'w') as f:
        json.dump({
            "video_name": track_data['video_name'],
            "original_fps": track_data.get('original_fps', 30.0),
            "tracks": pose_results
        }, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Robust Pose Extraction")
    parser.add_argument("--model", type=str, default="models/rtmpose-s-c54166.onnx")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tracks_dir", type=str, default="data/tracks")
    parser.add_argument("--video_dir", type=str, default="data/videos")
    parser.add_argument("--output_dir", type=str, default="data/poses")
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.7, help="Smoothing factor (0.0-1.0)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        exit(1)

    inferencer = RTMPoseInferencer(args.model, device=args.device)

    # Mapping between track categories and video directory names
    category_map = {
        "Violence": "Violence",
        "NonViolence": "NonViolence"
    }

    for category, v_subdir in category_map.items():
        t_dir = os.path.join(args.tracks_dir, category)
        v_dir = os.path.join(args.video_dir, v_subdir)
        o_dir = os.path.join(args.output_dir, category)
        
        if not os.path.exists(t_dir): continue
        os.makedirs(o_dir, exist_ok=True)
        
        for tf in sorted(os.listdir(t_dir)):
            if not tf.endswith(".json"): continue
            
            video_name = os.path.splitext(tf)[0]
            video_path = next((os.path.join(v_dir, f"{video_name}{ext}") 
                              for ext in ['.mp4', '.avi', '.mov', '.mkv'] 
                              if os.path.exists(os.path.join(v_dir, f"{video_name}{ext}"))), None)
            
            if not video_path: continue
            
            with open(os.path.join(t_dir, tf), 'r') as f:
                track_data = json.load(f)
            
            process_video_poses(video_path, track_data, inferencer, os.path.join(o_dir, tf), args.conf_thresh, args.alpha)

    print("\nStage 2: Refined Pose Extraction complete.")
