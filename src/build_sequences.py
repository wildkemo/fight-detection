import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class FeatureExtractor:
    """
    Converts a 60-frame window of (Person A, Person B)
    into a stable temporal feature representation.
    """

    def __init__(self, sequence_length=60):
        self.seq_len = sequence_length

        # COCO keypoints
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.WRIST_L = 9
        self.WRIST_R = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12

        self.TORSO_JOINTS = [5, 6, 11, 12]
        self.UPPER_BODY = [0, 5, 6, 7, 8, 9, 10]
        self.LOWER_BODY = [11, 12, 13, 14, 15, 16]

    # -------------------------
    # Normalization
    # -------------------------
    def normalize_skeletons(self, kpts_A, kpts_B):

        pelvis_A = (kpts_A[:, self.LEFT_HIP, :2] + kpts_A[:, self.RIGHT_HIP, :2]) / 2.0
        pelvis_B = (kpts_B[:, self.LEFT_HIP, :2] + kpts_B[:, self.RIGHT_HIP, :2]) / 2.0

        midpoint = (pelvis_A + pelvis_B) / 2.0
        mid = midpoint[:, None, :]

        norm_A = kpts_A[:, :, :2] - mid
        norm_B = kpts_B[:, :, :2] - mid

        def get_scale(kpts):
            coords = kpts[:, self.TORSO_JOINTS, :2]
            weights = kpts[:, self.TORSO_JOINTS, 2:3]

            torso_mean = np.sum(coords * weights, axis=1, keepdims=True) / (
                np.sum(weights, axis=1, keepdims=True) + 1e-6
            )

            dist = np.sqrt(np.sum((coords - torso_mean) ** 2, axis=2))
            # Mask distances for low-confidence joints to avoid (0,0) pulling the scale
            dist = dist * weights[:, :, 0]
            # Average over only the visible joints
            scale = np.sum(dist, axis=1) / (np.sum(weights[:, :, 0], axis=1) + 1e-6)
            return scale + 1e-6

        scale_A = get_scale(kpts_A)
        scale_B = get_scale(kpts_B)

        scale = np.maximum(scale_A, scale_B)[:, None, None]

        return norm_A / scale, norm_B / scale

    # -------------------------
    # Motion
    # -------------------------
    def compute_motion(self, kpts):
        vel = np.diff(kpts, axis=0, prepend=kpts[:1])
        
        # Split energy into upper and lower body to distinguish fighting from walking
        upper_energy = np.sum(vel[:, self.UPPER_BODY] ** 2, axis=(1, 2))[:, None]
        lower_energy = np.sum(vel[:, self.LOWER_BODY] ** 2, axis=(1, 2))[:, None]
        
        return vel, upper_energy, lower_energy

    # -------------------------
    # Interaction
    # -------------------------
    def compute_interaction(self, norm_A, norm_B, vel_A, vel_B):

        pelvis_A = (norm_A[:, self.LEFT_HIP] + norm_A[:, self.RIGHT_HIP]) / 2.0
        pelvis_B = (norm_B[:, self.LEFT_HIP] + norm_B[:, self.RIGHT_HIP]) / 2.0

        dist = np.linalg.norm(pelvis_A - pelvis_B, axis=1)[:, None]
        closing = np.diff(dist, axis=0, prepend=dist[:1])

        rel_vel = vel_A - vel_B

        def j_dist(p1, p2):
            return np.linalg.norm(p1 - p2, axis=1)[:, None]

        crit = np.concatenate([
            j_dist(norm_A[:, self.WRIST_L], norm_B[:, self.NOSE]),
            j_dist(norm_A[:, self.WRIST_R], norm_B[:, self.NOSE]),
            j_dist(norm_B[:, self.WRIST_L], norm_A[:, self.NOSE]),
            j_dist(norm_B[:, self.WRIST_R], norm_A[:, self.NOSE]),
        ], axis=1)

        return dist, closing, rel_vel, crit

    # -------------------------
    # Extract features
    # -------------------------
    def extract(self, kpts_A, kpts_B):

        # ---- causal smoothing (no roll bug) ----
        A = kpts_A.copy()
        B = kpts_B.copy()

        A[1:] = (A[1:] + A[:-1]) / 2.0
        B[1:] = (B[1:] + B[:-1]) / 2.0

        norm_A, norm_B = self.normalize_skeletons(A, B)

        vA, ueA, leA = self.compute_motion(norm_A)
        vB, ueB, leB = self.compute_motion(norm_B)

        dist, closing, rel_v, crit = self.compute_interaction(
            norm_A, norm_B, vA, vB
        )

        conf_A = kpts_A[:, :, 2]
        conf_B = kpts_B[:, :, 2]

        features = []
        for i in range(self.seq_len):
            feat = np.concatenate([
                norm_A[i].flatten(),
                norm_B[i].flatten(),
                vA[i].flatten(),
                vB[i].flatten(),
                ueA[i],
                leA[i],
                ueB[i],
                leB[i],
                dist[i],
                closing[i],
                rel_v[i].flatten(),
                crit[i],
                conf_A[i].flatten(),
                conf_B[i].flatten()
            ])
            features.append(feat)

        return np.array(features)



# =========================================================
# Dataset builder
# =========================================================
def build_sequences(poses_dir, output_dir, proximity_thresh=0.4):

    extractor = FeatureExtractor()
    X, y = [], []

    FEATURE_DIM = None

    for label_str, label_val in [("Violence", 1), ("NonViolence", 0)]:

        folder = os.path.join(poses_dir, label_str)
        if not os.path.exists(folder):
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".json")]

        for f in tqdm(files, desc=f"Building {label_str}"):

            with open(os.path.join(folder, f), "r") as fp:
                data = json.load(fp)

            frame_data = {}

            for tid, frames in data["tracks"].items():
                for fr in frames:
                    idx = fr["frame_idx"]
                    frame_data.setdefault(idx, []).append({
                        "tid": tid,
                        "kpts": np.array(fr["keypoints"])
                    })

            sorted_frames = sorted(frame_data.keys())
            if not sorted_frames:
                continue

            buffers = {}

            for idx in sorted_frames:
                people = frame_data[idx]
                present = set()

                for i in range(len(people)):
                    for j in range(i + 1, len(people)):

                        p1, p2 = people[i], people[j]

                        # mean confidence filter
                        if np.mean(p1["kpts"][:, 2]) < 0.4 or np.mean(p2["kpts"][:, 2]) < 0.4:
                            continue

                        p1_center = (p1["kpts"][11, :2] + p1["kpts"][12, :2]) / 2
                        p2_center = (p2["kpts"][11, :2] + p2["kpts"][12, :2]) / 2

                        # Diagonal normalization for proximity check
                        # If the JSONs are in pixels, we should use a reference resolution or diagonal.
                        # Assuming 1080p as reference for the 0.4 threshold.
                        scale = np.linalg.norm([1920, 1080])
                        d = np.linalg.norm(p1_center - p2_center) / scale

                        if d > proximity_thresh:
                            continue

                        pair_id = tuple(sorted([p1["tid"], p2["tid"]]))
                        present.add(pair_id)

                        if pair_id not in buffers:
                            buffers[pair_id] = {
                                "A_id": p1["tid"],
                                "B_id": p2["tid"],
                                "A": [],
                                "B": [],
                                "last": idx
                            }

                        buf = buffers[pair_id]

                        if idx - buf["last"] > 2:
                            buf["A"], buf["B"] = [], []

                        buf["last"] = idx

                        if p1["tid"] == buf["A_id"]:
                            A, B = p1["kpts"], p2["kpts"]
                        else:
                            A, B = p2["kpts"], p1["kpts"]

                        # swap protection (fixed)
                        if len(buf["A"]) > 0:
                            prev_A = buf["A"][-1]
                            prev_B = buf["B"][-1]

                            prev_A_c = (prev_A[11, :2] + prev_A[12, :2]) / 2
                            curr_A_c = (A[11, :2] + A[12, :2]) / 2

                            prev_B_c = (prev_B[11, :2] + prev_B[12, :2]) / 2
                            curr_B_c = (B[11, :2] + B[12, :2]) / 2

                            jump_A = np.linalg.norm(curr_A_c - prev_A_c) / scale
                            jump_B = np.linalg.norm(curr_B_c - prev_B_c) / scale

                            if jump_A > 0.3 or jump_B > 0.3:
                                buf["A"], buf["B"] = [], []

                        buf["A"].append(A)
                        buf["B"].append(B)

                        while len(buf["A"]) >= 60:

                            feat = extractor.extract(
                                np.array(buf["A"][:60]),
                                np.array(buf["B"][:60])
                            )

                            if FEATURE_DIM is None:
                                FEATURE_DIM = feat.shape[1]

                            if feat.shape != (60, FEATURE_DIM):
                                continue

                            X.append(feat)
                            y.append(label_val)

                            buf["A"] = buf["A"][30:]
                            buf["B"] = buf["B"][30:]

                for pid in list(buffers.keys()):
                    if pid not in present:
                        del buffers[pid]

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    print("Done")
    print(X.shape, "feature dim:", X.shape[2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses_dir", default="data/poses")
    parser.add_argument("--output_dir", default="data/sequences")
    parser.add_argument("--proximity", type=float, default=0.4)

    args = parser.parse_args()
    build_sequences(args.poses_dir, args.output_dir, args.proximity)