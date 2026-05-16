"""
inference.py

Real-time Fight Detection System

Pipeline:
Frame
    -> YOLOv8 Detection + ByteTrack
    -> Pairwise Proximity Filtering
    -> RTMPose ONNX Pose Estimation
    -> Temporal Pose Smoothing
    -> Feature Extraction
    -> 36-frame Sequence Building
    -> TCN TFLite Inference
    -> Temporal Decision Smoothing
    -> Fight Alert

Architecture matches the training pipeline exactly.

Author: Taher
"""

import os
import cv2
import math
import time
import argparse
import numpy as np
import tensorflow as tf
import onnxruntime as ort

from ultralytics import YOLO
from collections import defaultdict, deque

# ============================================================
# CONFIG
# ============================================================

SEQUENCE_LENGTH = 36

CONF_THRESH = 0.3
POSE_SMOOTH_ALPHA = 0.7

PAIR_DISTANCE_THRESHOLD = 0.15

TEMPORAL_WINDOW = 10
FIGHT_THRESHOLD = 0.72
TRIGGER_COUNT = 6

DISPLAY = True

# ============================================================
# FEATURE EXTRACTOR
# ============================================================


class FeatureExtractor:

    def __init__(self, sequence_length=36):

        self.seq_len = sequence_length

        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.WRIST_L = 9
        self.WRIST_R = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12

        self.TORSO_JOINTS = [5, 6, 11, 12]

    def normalize_skeletons(self, kpts_A, kpts_B):

        pelvis_A = (
            kpts_A[:, self.LEFT_HIP, :2]
            + kpts_A[:, self.RIGHT_HIP, :2]
        ) / 2.0

        pelvis_B = (
            kpts_B[:, self.LEFT_HIP, :2]
            + kpts_B[:, self.RIGHT_HIP, :2]
        ) / 2.0

        midpoint = (pelvis_A + pelvis_B) / 2.0
        mid = midpoint[:, None, :]

        norm_A = kpts_A[:, :, :2] - mid
        norm_B = kpts_B[:, :, :2] - mid

        def get_scale(kpts):

            coords = kpts[:, self.TORSO_JOINTS, :2]
            weights = kpts[:, self.TORSO_JOINTS, 2:3]

            torso_mean = np.sum(
                coords * weights,
                axis=1,
                keepdims=True
            ) / (np.sum(weights, axis=1, keepdims=True) + 1e-6)

            dist = np.sqrt(
                np.sum((coords - torso_mean) ** 2, axis=2)
            )

            return np.mean(dist, axis=1) + 1e-6

        scale_A = get_scale(kpts_A)
        scale_B = get_scale(kpts_B)

        scale = np.maximum(scale_A, scale_B)[:, None, None]

        return norm_A / scale, norm_B / scale

    def compute_motion(self, kpts):

        vel = np.diff(kpts, axis=0, prepend=kpts[:1])
        energy = np.sum(vel ** 2, axis=(1, 2))[:, None]

        return vel, energy

    def compute_interaction(self, norm_A, norm_B, vel_A, vel_B):

        pelvis_A = (
            norm_A[:, self.LEFT_HIP]
            + norm_A[:, self.RIGHT_HIP]
        ) / 2.0

        pelvis_B = (
            norm_B[:, self.LEFT_HIP]
            + norm_B[:, self.RIGHT_HIP]
        ) / 2.0

        dist = np.linalg.norm(
            pelvis_A - pelvis_B,
            axis=1
        )[:, None]

        closing = np.diff(
            dist,
            axis=0,
            prepend=dist[:1]
        )

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

    def extract(self, kpts_A, kpts_B):

        A = kpts_A.copy()
        B = kpts_B.copy()

        A[1:] = (A[1:] + A[:-1]) / 2.0
        B[1:] = (B[1:] + B[:-1]) / 2.0

        norm_A, norm_B = self.normalize_skeletons(A, B)

        vA, eA = self.compute_motion(norm_A)
        vB, eB = self.compute_motion(norm_B)

        dist, closing, rel_v, crit = self.compute_interaction(
            norm_A,
            norm_B,
            vA,
            vB
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
                eA[i],
                eB[i],
                dist[i],
                closing[i],
                rel_v[i].flatten(),
                crit[i],
                conf_A[i].flatten(),
                conf_B[i].flatten()
            ])

            features.append(feat)

        return np.array(features, dtype=np.float32)


# ============================================================
# RTMPOSE
# ============================================================


class RTMPoseInferencer:

    def __init__(self, model_path):

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [
            o.name for o in self.session.get_outputs()
        ]

        shape = self.session.get_inputs()[0].shape

        self.model_h = shape[2]
        self.model_w = shape[3]

        self.mean = np.array(
            [123.675, 116.28, 103.53],
            dtype=np.float32
        ).reshape(1, 1, 3)

        self.std = np.array(
            [58.395, 57.12, 57.375],
            dtype=np.float32
        ).reshape(1, 1, 3)

    def preprocess(self, frame, bbox):

        x1, y1, x2, y2 = bbox

        crop = frame[int(y1):int(y2), int(x1):int(x2)]

        if crop.size == 0:
            return None

        crop = cv2.resize(
            crop,
            (self.model_w, self.model_h)
        )

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        crop = (
            crop.astype(np.float32) - self.mean
        ) / self.std

        crop = crop.transpose(2, 0, 1)

        return crop[None]

    def decode(self, outputs):

        simcc_x = outputs[0]
        simcc_y = outputs[1]

        x = np.argmax(simcc_x, axis=2)
        y = np.argmax(simcc_y, axis=2)

        x_scores = np.max(simcc_x, axis=2)
        y_scores = np.max(simcc_y, axis=2)

        scores = (x_scores + y_scores) / 2.0

        results = np.stack([
            x.astype(np.float32),
            y.astype(np.float32),
            scores
        ], axis=2)

        return results

    def infer(self, frame, bbox):

        x1, y1, x2, y2 = bbox

        inp = self.preprocess(frame, bbox)

        if inp is None:
            return None

        outputs = self.session.run(
            self.output_names,
            {self.input_name: inp}
        )

        kpts = self.decode(outputs)[0]

        # restore local coords
        kpts[:, 0] = x1 + (
            kpts[:, 0] / self.model_w
        ) * (x2 - x1)

        kpts[:, 1] = y1 + (
            kpts[:, 1] / self.model_h
        ) * (y2 - y1)

        h, w = frame.shape[:2]

        kpts[:, 0] /= w
        kpts[:, 1] /= h

        return kpts


# ============================================================
# TFLITE CLASSIFIER
# ============================================================


class TCNClassifier:

    def __init__(self, model_path):

        self.interpreter = tf.lite.Interpreter(
            model_path=model_path
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter.get_input_details()
        )

        self.output_details = (
            self.interpreter.get_output_details()
        )

    def predict(self, sequence):

        sequence = np.expand_dims(
            sequence.astype(np.float32),
            axis=0
        )

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            sequence
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        return float(output[0][0])


# ============================================================
# HELPERS
# ============================================================


def bbox_center(box):

    x1, y1, x2, y2 = box

    return np.array([
        (x1 + x2) / 2,
        (y1 + y2) / 2
    ])


def pair_distance(boxA, boxB, frame_shape):

    h, w = frame_shape[:2]

    scale = np.linalg.norm([w, h])

    return np.linalg.norm(
        bbox_center(boxA) - bbox_center(boxB)
    ) / scale


def temporal_decision(history):

    high = [x for x in history if x >= FIGHT_THRESHOLD]

    return len(high) >= TRIGGER_COUNT


# ============================================================
# MAIN
# ============================================================


def run_inference(args):

    detector = YOLO(args.yolo)

    pose_model = RTMPoseInferencer(args.rtmpose)

    classifier = TCNClassifier(args.tcn)

    extractor = FeatureExtractor()

    cap = cv2.VideoCapture(args.source)

    pair_memory = defaultdict(lambda: {
        "A": deque(maxlen=SEQUENCE_LENGTH),
        "B": deque(maxlen=SEQUENCE_LENGTH),
        "scores": deque(maxlen=TEMPORAL_WINDOW),
        "fight": False,
        "last_seen": 0,
        "prob": 0.0
    })

    pose_memory = {}

    frame_idx = 0
    prev_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ====================================================
        # DETECTION + TRACKING
        # ====================================================

        results = detector.track(
            source=frame,
            persist=True,
            classes=[0],
            tracker=args.tracker,
            conf=args.conf,
            verbose=False
        )

        people = []

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, tid in zip(boxes, ids):

                people.append({
                    "id": tid,
                    "bbox": box
                })

        # ====================================================
        # INTERACTION FILTERING
        # ====================================================

        valid_pairs = []

        for i in range(len(people)):

            for j in range(i + 1, len(people)):

                p1 = people[i]
                p2 = people[j]

                dist = pair_distance(
                    p1["bbox"],
                    p2["bbox"],
                    frame.shape
                )

                if dist < PAIR_DISTANCE_THRESHOLD:

                    pair_id = tuple(
                        sorted([p1["id"], p2["id"]])
                    )

                    valid_pairs.append(
                        (pair_id, p1, p2)
                    )

        # ====================================================
        # POSE
        # ====================================================

        current_poses = {}

        needed_ids = set()

        for _, p1, p2 in valid_pairs:

            needed_ids.add(p1["id"])
            needed_ids.add(p2["id"])

        for p in people:

            tid = p["id"]

            if tid not in needed_ids:
                continue

            kpts = pose_model.infer(
                frame,
                p["bbox"]
            )

            if kpts is None:
                continue

            if tid in pose_memory:

                prev = pose_memory[tid]

                smoothed = kpts.copy()

                for j in range(len(kpts)):

                    x, y, c = kpts[j]
                    px, py, _ = prev[j]

                    if c < CONF_THRESH:

                        smoothed[j, 0] = px
                        smoothed[j, 1] = py

                    else:

                        smoothed[j, 0] = (
                            POSE_SMOOTH_ALPHA * x
                            + (1 - POSE_SMOOTH_ALPHA) * px
                        )

                        smoothed[j, 1] = (
                            POSE_SMOOTH_ALPHA * y
                            + (1 - POSE_SMOOTH_ALPHA) * py
                        )

                kpts = smoothed

            pose_memory[tid] = kpts.copy()

            current_poses[tid] = kpts

        # ====================================================
        # BUILD PAIR SEQUENCES
        # ====================================================

        for pair_id, p1, p2 in valid_pairs:

            idA, idB = pair_id

            if idA not in current_poses:
                continue

            if idB not in current_poses:
                continue

            kpts_A = current_poses[idA]
            kpts_B = current_poses[idB]

            mem = pair_memory[pair_id]

            mem["A"].append(kpts_A)
            mem["B"].append(kpts_B)

            mem["last_seen"] = frame_idx

            # ================================================
            # INFERENCE
            # ================================================

            if len(mem["A"]) == SEQUENCE_LENGTH:

                seq_A = np.array(mem["A"])
                seq_B = np.array(mem["B"])

                features = extractor.extract(
                    seq_A,
                    seq_B
                )

                prob = classifier.predict(features)

                mem["prob"] = prob

                mem["scores"].append(prob)

                mem["fight"] = temporal_decision(
                    mem["scores"]
                )

        # ====================================================
        # CLEAN OLD PAIRS
        # ====================================================

        stale = []

        for pair_id, mem in pair_memory.items():

            if frame_idx - mem["last_seen"] > 20:
                stale.append(pair_id)

        for s in stale:
            del pair_memory[s]

        # ====================================================
        # VISUALIZATION
        # ====================================================

        for pair_id, mem in pair_memory.items():

            if not mem["fight"]:
                continue

            idA, idB = pair_id

            for p in people:

                if p["id"] in [idA, idB]:

                    x1, y1, x2, y2 = map(
                        int,
                        p["bbox"]
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        frame,
                        f"FIGHT {mem['prob']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

        # ====================================================
        # FPS
        # ====================================================

        current = time.time()

        fps = 1.0 / (current - prev_time)

        prev_time = current

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        if DISPLAY:

            cv2.imshow(
                "Fight Detection",
                frame
            )

            key = cv2.waitKey(1)

            if key == 27:
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        default=0
    )

    parser.add_argument(
        "--yolo",
        type=str,
        default="models/yolov8n.pt"
    )

    parser.add_argument(
        "--rtmpose",
        type=str,
        default="models/rtmpose.onnx"
    )

    parser.add_argument(
        "--tcn",
        type=str,
        default="models/tcn_model.tflite"
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.3
    )

    args = parser.parse_args()

    run_inference(args)