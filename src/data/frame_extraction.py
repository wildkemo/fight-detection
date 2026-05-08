"""
Extract uniformly sampled frames from videos as JPEG files.

Layout (compatible with ``human_detection.py``):

  FRAMES_DIR / <class_name> / <video_stem> / frame_00000.jpg ...

Requires: opencv-python, numpy, tqdm

Run from project root::

  PYTHONPATH=. python src/data/frame_extraction.py

By default frames are saved at **native resolution** (good for downstream YOLO).
Use ``--resize`` to save at ``FRAME_WIDTH`` x ``FRAME_HEIGHT`` instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import CLASS_NAMES  # noqa: E402
from src.utils.config import (  # noqa: E402
    DATASET_DIR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FRAMES_DIR,
    SEQUENCE_LENGTH,
)

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}


def get_frame_indices(total_frames: int, sequence_length: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)

    if total_frames <= sequence_length:
        return np.linspace(0, total_frames - 1, total_frames, dtype=int)

    return np.linspace(0, total_frames - 1, sequence_length, dtype=int)


def extract_video_to_folder(
    video_path: Path,
    out_folder: Path,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    resize_wh: tuple[int, int] | None = None,
    jpeg_quality: int = 92,
) -> bool:
    """
    Sample ``sequence_length`` frames (uniformly over the clip), optionally
    resize, write ``frame_%05d.jpg`` under ``out_folder``.
    """
    frames: list[np.ndarray] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = get_frame_indices(total_frames, sequence_length)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()

        if not success:
            continue

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        frames = [
            np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8),
        ]
    while len(frames) < sequence_length:
        frames.append(frames[-1].copy())

    out_folder.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        if resize_wh is not None:
            tw, th = resize_wh
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        dest = out_folder / f"frame_{i:05d}.jpg"
        cv2.imwrite(
            str(dest),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )

    return True


def extract_frames(
    video_path: str | Path,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    resize_wh: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Sample frames into a single array (T, H, W, 3) uint8 BGR — no disk write.
    """
    video_path = Path(video_path)
    frames: list[np.ndarray] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = get_frame_indices(total_frames, sequence_length)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if success:
            frames.append(frame)

    cap.release()

    if not frames:
        frames = [np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)]
    while len(frames) < sequence_length:
        frames.append(frames[-1].copy())

    out_list: list[np.ndarray] = []
    for frame in frames:
        if resize_wh is not None:
            tw, th = resize_wh
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        out_list.append(frame)

    return np.stack(out_list, axis=0)


def iter_videos(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []
    out: list[Path] = []
    for entry in sorted(class_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in _VIDEO_SUFFIXES:
            out.append(entry)
    return out


def run_frame_extraction_pipeline(
    *,
    dataset_dir: Path | str = DATASET_DIR,
    frames_dir: Path | str = FRAMES_DIR,
    class_names: list[str] | None = None,
    sequence_length: int = SEQUENCE_LENGTH,
    resize_wh: tuple[int, int] | None = None,
) -> dict:
    """
    dataset_dir /<class>/<video files>
    -> frames_dir /<class>/<video stem>/frame_*.jpg
    """
    dataset_dir = Path(dataset_dir)
    frames_dir = Path(frames_dir)
    labels = list(class_names) if class_names is not None else list(CLASS_NAMES)

    stats = {"classes": 0, "videos": 0, "failed": 0}

    os.chdir(_ROOT)

    for cls in labels:
        in_cls = dataset_dir / cls
        videos = iter_videos(in_cls)
        if not videos:
            print(f"Skip (no videos): {in_cls}")
            continue

        stats["classes"] += 1

        print(f"\nProcessing class {cls}: {len(videos)} videos\n")

        for video_path in tqdm(videos):
            stem = video_path.stem
            out_folder = frames_dir / cls / stem
            ok = extract_video_to_folder(
                video_path,
                out_folder,
                sequence_length=sequence_length,
                resize_wh=resize_wh,
            )
            stats["videos"] += 1
            if not ok:
                stats["failed"] += 1

    print("Frame extraction:", stats)
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract video frames as JPEG folders.")
    p.add_argument(
        "--dataset",
        default=str(_ROOT / DATASET_DIR),
        help="Dataset root containing class folders of videos.",
    )
    p.add_argument(
        "--out",
        default=str(_ROOT / FRAMES_DIR),
        help=f"Output root (default: project/{FRAMES_DIR}).",
    )
    p.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Number of frames sampled per video.",
    )
    p.add_argument(
        "--resize",
        action="store_true",
        help=f"Resize each frame to {FRAME_WIDTH}x{FRAME_HEIGHT} before saving.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    resize_wh = (FRAME_WIDTH, FRAME_HEIGHT) if args.resize else None
    run_frame_extraction_pipeline(
        dataset_dir=args.dataset,
        frames_dir=args.out,
        sequence_length=args.sequence_length,
        resize_wh=resize_wh,
    )


if __name__ == "__main__":
    main()
