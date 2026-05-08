"""
Person-centric crops from extracted frame folders using YOLO.

Expects layout (same as typical frame extraction outputs):
  FRAMES_DIR / <class_name> / <video_id> / frame_*.jpg

Writes mirror layout to HUMAN_CROPS_DIR.

Requires:
  pip install ultralytics

Run from project root:
  PYTHONPATH=. python src/data/human_detection.py

Or:
  PYTHONPATH=. python -m src.data.human_detection
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np

# Allow `python src/data/human_detection.py` without installing the package.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import CLASS_NAMES  # noqa: E402
from src.utils.config import FRAME_HEIGHT, FRAME_WIDTH, FRAMES_DIR, HUMAN_CROPS_DIR  # noqa: E402


try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - optional dependency
    YOLO = None
    _ULTRAIMPORT_ERR = exc
else:
    _ULTRAIMPORT_ERR = None

# COCO: class 0 is "person"
PERSON_CLASS_ID = 0
_FRAME_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

FallbackMode = Literal["resize_full_frame", "black"]


def _require_ultralytics() -> None:
    if YOLO is None:
        raise ImportError(
            "ultralytics is required for human detection. "
            "Install with: pip install ultralytics"
        ) from _ULTRAIMPORT_ERR


def _list_frames(video_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in _FRAME_GLOBS:
        files.extend(video_dir.glob(pattern))
    return sorted({p.resolve(): p for p in files}.values(), key=lambda p: p.name)


def expand_xyxy(xyxy: "object", ih: int, iw: int, pad_frac: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, xyxy)
    w_box = max(x2 - x1, 1.0)
    h_box = max(y2 - y1, 1.0)
    pad_x = w_box * pad_frac
    pad_y = h_box * pad_frac
    x1i = max(0, int(round(x1 - pad_x)))
    y1i = max(0, int(round(y1 - pad_y)))
    x2i = min(iw, int(round(x2 + pad_x)))
    y2i = min(ih, int(round(y2 + pad_y)))
    x2i = max(x1i + 1, x2i)
    y2i = max(y1i + 1, y2i)
    return x1i, y1i, x2i, y2i


def pick_largest_person_box(result) -> Optional["object"]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best_area = -1.0
    best_xyxy = None
    for i in range(len(boxes)):
        if int(boxes.cls[i].item()) != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        if area > best_area:
            best_area = area
            best_xyxy = boxes.xyxy[i]
    return best_xyxy


def detect_and_crop_frame(
    frame_bgr,
    model,
    *,
    conf: float = 0.25,
    pad_frac: float = 0.15,
    target_wh: tuple[int, int] | None = None,
    fallback_mode: FallbackMode = "resize_full_frame",
    device: str = "cpu",
):
    """Return cropped (or fallback) BGR frame, resized to ``target_wh`` if set."""

    ih, iw = frame_bgr.shape[:2]

    predict_kwargs = dict(
        source=frame_bgr,
        conf=conf,
        classes=[PERSON_CLASS_ID],
        verbose=False,
        device=device,
    )

    results = model.predict(**predict_kwargs)
    if not results:
        box_xyxy = None
    else:
        box_xyxy = pick_largest_person_box(results[0])

    if box_xyxy is not None:
        x1, y1, x2, y2 = expand_xyxy(box_xyxy, ih, iw, pad_frac)
        crop = frame_bgr[y1:y2, x1:x2].copy()
    else:
        if fallback_mode == "black":
            crop = np.zeros((ih, iw, 3), dtype=frame_bgr.dtype)
        else:
            crop = frame_bgr.copy()

    if target_wh is not None:
        tw, th = target_wh
        crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

    return crop


def process_frame_file(
    model,
    frame_path: Path,
    save_path: Path,
    *,
    conf: float,
    pad_frac: float,
    target_wh: tuple[int, int] | None,
    fallback_mode: FallbackMode,
    device: str,
) -> bool:

    img = cv2.imread(str(frame_path))
    if img is None:
        return False
    crop = detect_and_crop_frame(
        img,
        model,
        conf=conf,
        pad_frac=pad_frac,
        target_wh=target_wh,
        fallback_mode=fallback_mode,
        device=device,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(save_path), crop)


def run_human_crop_pipeline(
    frames_root: str | Path = FRAMES_DIR,
    out_root: str | Path = HUMAN_CROPS_DIR,
    model_weights: str = "yolov8n.pt",
    *,
    conf: float = 0.25,
    pad_frac: float = 0.15,
    target_wh: tuple[int, int] | None = None,
    fallback_mode: FallbackMode = "resize_full_frame",
    device: str = "cpu",
    class_names: Optional[list[str]] = None,

) -> dict:

    _require_ultralytics()
    model = YOLO(model_weights)

    frames_root = Path(frames_root)
    out_root = Path(out_root)
    labels = list(class_names) if class_names is not None else list(CLASS_NAMES)

    stats = {"videos": 0, "frames_ok": 0, "frames_fail_read": 0}

    if target_wh is None:
        tw, th = FRAME_WIDTH, FRAME_HEIGHT
    else:
        tw, th = target_wh

    for class_name in labels:
        class_in = frames_root / class_name
        if not class_in.is_dir():
            print(f"Skip missing class folder: {class_in}")
            continue

        for video_dir in sorted(p for p in class_in.iterdir() if p.is_dir()):
            stats["videos"] += 1
            rel = video_dir.relative_to(class_in)
            out_video = out_root / class_name / rel
            for frame_path in _list_frames(video_dir):
                out_name = frame_path.name
                if frame_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    out_name = frame_path.stem + ".jpg"
                save_path = out_video / out_name
                ok = process_frame_file(
                    model,
                    frame_path,
                    save_path,
                    conf=conf,
                    pad_frac=pad_frac,
                    target_wh=(tw, th),
                    fallback_mode=fallback_mode,
                    device=device,
                )
                if ok:
                    stats["frames_ok"] += 1
                else:
                    stats["frames_fail_read"] += 1

    print("Human crop pipeline done:", stats)
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO person crops from FRAMES_DIR tree.")
    p.add_argument(
        "--frames-root",
        default=FRAMES_DIR,
        help="Root with class/video folders of frames.",
    )
    p.add_argument(
        "--out-root",
        default=HUMAN_CROPS_DIR,
        help="Output root (mirror structure).",
    )
    p.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="YOLO weights path or model name (e.g. yolov8n.pt).",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--pad-frac", type=float, default=0.15)
    p.add_argument("--width", type=int, default=FRAME_WIDTH)
    p.add_argument("--height", type=int, default=FRAME_HEIGHT)
    p.add_argument(
        "--device",
        default="cpu",
        help="Ultralytics device string, e.g. cpu or 0.",
    )
    p.add_argument(
        "--fallback",
        choices=("resize_full_frame", "black"),
        default="resize_full_frame",
    )
    return p.parse_args()


def main() -> None:

    _require_ultralytics()
    os.chdir(_ROOT)

    args = _parse_args()
    run_human_crop_pipeline(
        frames_root=args.frames_root,
        out_root=args.out_root,
        model_weights=args.weights,
        conf=args.conf,
        pad_frac=args.pad_frac,
        target_wh=(args.width, args.height),
        fallback_mode=args.fallback,
        device=args.device,
    )


if __name__ == "__main__":
    main()
