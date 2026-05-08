"""
Region-of-interest refinement on **human-centric** crops (from YOLO).

Human detection already removes most of the scene. This step reduces **static
context** inside the crop (walls, floor, parked shapes) by focusing on pixels
that **change over time** — a cheap proxy for limbs and interaction, not yet
a violence classifier.

Expects the same tree as ``human_detection`` output::

  HUMAN_CROPS_DIR / <class> / <video_id> / frame_*.jpg

Writes::

  ROI_CROPS_DIR / <class> / <video_id> / frame_*.jpg

Run from project root::

  PYTHONPATH=. python src/data/crop_roi.py

Requires only OpenCV + NumPy (CPU-friendly).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import CLASS_NAMES  # noqa: E402
from src.utils.config import (  # noqa: E402
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HUMAN_CROPS_DIR,
    ROI_CROPS_DIR,
)

_FRAME_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def _list_frames(video_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in _FRAME_GLOBS:
        files.extend(video_dir.glob(pattern))
    return sorted({p.resolve(): p for p in files}.values(), key=lambda p: p.name)


def _to_gray_blur(
    bgr: np.ndarray,
    blur_ksize: int,
) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = max(3, blur_ksize | 1)
    g = cv2.GaussianBlur(g, (k, k), 0)
    return g


def motion_mask(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    diff_threshold: int,
    morph_kernel: int,
    dilate_iterations: int,
) -> np.ndarray:
    d = cv2.absdiff(gray_prev, gray_curr)
    _, mask = cv2.threshold(d, diff_threshold, 255, cv2.THRESH_BINARY)
    mk = max(3, morph_kernel | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
    return mask


def largest_motion_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) inclusive-exclusive in OpenCV slice style, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < 1.0:
        return None
    x, y, w, h = cv2.boundingRect(best)
    return x, y, x + w, y + h


def expand_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    ih: int,
    iw: int,
    pad_frac: float,
    min_side_frac: float,
) -> tuple[int, int, int, int]:
    w_box = max(x2 - x1, 1)
    h_box = max(y2 - y1, 1)
    px = int(w_box * pad_frac)
    py = int(h_box * pad_frac)
    x1e = max(0, x1 - px)
    y1e = max(0, y1 - py)
    x2e = min(iw, x2 + px)
    y2e = min(ih, y2 + py)

    min_w = max(2, int(iw * min_side_frac))
    min_h = max(2, int(ih * min_side_frac))

    cur_w = x2e - x1e
    cur_h = y2e - y1e
    cx = (x1e + x2e) // 2
    cy = (y1e + y2e) // 2

    if cur_w < min_w:
        half = min_w // 2
        x1e = max(0, cx - half)
        x2e = min(iw, x1e + min_w)
        if x2e - x1e < min_w:
            x1e = max(0, x2e - min_w)

    cur_h = y2e - y1e
    cy = (y1e + y2e) // 2
    if cur_h < min_h:
        half = min_h // 2
        y1e = max(0, cy - half)
        y2e = min(ih, y1e + min_h)
        if y2e - y1e < min_h:
            y1e = max(0, y2e - min_h)

    x2e = max(x1e + 1, min(iw, x2e))
    y2e = max(y1e + 1, min(ih, y2e))
    return x1e, y1e, x2e, y2e


def roi_xyxy_from_motion(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    ih: int,
    iw: int,
    *,
    diff_threshold: int,
    morph_kernel: int,
    dilate_iterations: int,
    pad_frac: float,
    min_motion_area_ratio: float,
    min_side_frac: float,
) -> Optional[tuple[int, int, int, int]]:
    mask = motion_mask(
        gray_prev,
        gray_curr,
        diff_threshold=diff_threshold,
        morph_kernel=morph_kernel,
        dilate_iterations=dilate_iterations,
    )
    motion_pixels = float(cv2.countNonZero(mask))
    if motion_pixels < min_motion_area_ratio * (ih * iw):
        return None
    bb = largest_motion_bbox(mask)
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    return expand_xyxy(x1, y1, x2, y2, ih, iw, pad_frac, min_side_frac)


def apply_roi_crop(
    img_bgr: np.ndarray,
    xyxy: Optional[tuple[int, int, int, int]],
    target_wh: tuple[int, int],
) -> np.ndarray:
    ih, iw = img_bgr.shape[:2]
    if xyxy is None:
        crop = img_bgr
    else:
        x1, y1, x2, y2 = xyxy
        crop = img_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, target_wh, interpolation=cv2.INTER_AREA)


def refine_video_folder(
    frame_paths: list[Path],
    out_dir: Path,
    *,
    diff_threshold: int = 18,
    blur_ksize: int = 5,
    morph_kernel: int = 5,
    dilate_iterations: int = 2,
    motion_pad_frac: float = 0.28,
    min_motion_area_ratio: float = 0.012,
    min_side_frac: float = 0.42,
    target_wh: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT),
    jpeg_quality: int = 92,
) -> tuple[int, int]:
    """Process one clip folder. Returns (frames_written, frames_failed_read)."""

    ok = 0
    fails = 0
    loaded: list[np.ndarray] = []

    out_dir.mkdir(parents=True, exist_ok=True)

    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            fails += 1
            loaded.append(None)
            continue
        loaded.append(img)
        ok += 1

    if not loaded or all(x is None for x in loaded):
        return 0, fails

    n = len(loaded)

    grays = []
    for im in loaded:
        if im is None:
            grays.append(None)
            continue
        grays.append(_to_gray_blur(im, blur_ksize))

    for i in range(n):
        out_name = frame_paths[i].name
        img = loaded[i]
        if img is None:
            continue

        ih, iw = img.shape[:2]
        roi: Optional[tuple[int, int, int, int]] = None

        if n == 1:
            roi = None
        elif i == 0 and grays[0] is not None and grays[1] is not None:
            roi = roi_xyxy_from_motion(
                grays[0],
                grays[1],
                ih,
                iw,
                diff_threshold=diff_threshold,
                morph_kernel=morph_kernel,
                dilate_iterations=dilate_iterations,
                pad_frac=motion_pad_frac,
                min_motion_area_ratio=min_motion_area_ratio,
                min_side_frac=min_side_frac,
            )
        elif i > 0 and grays[i - 1] is not None and grays[i] is not None:
            roi = roi_xyxy_from_motion(
                grays[i - 1],
                grays[i],
                ih,
                iw,
                diff_threshold=diff_threshold,
                morph_kernel=morph_kernel,
                dilate_iterations=dilate_iterations,
                pad_frac=motion_pad_frac,
                min_motion_area_ratio=min_motion_area_ratio,
                min_side_frac=min_side_frac,
            )

        crop = apply_roi_crop(img, roi, target_wh)
        dest = out_dir / out_name
        if dest.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            dest = dest.with_suffix(".jpg")
        cv2.imwrite(
            str(dest),
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )

    return ok, fails


def run_roi_crop_pipeline(
    input_root: str | Path = HUMAN_CROPS_DIR,
    output_root: str | Path = ROI_CROPS_DIR,
    *,
    diff_threshold: int = 18,
    blur_ksize: int = 5,
    morph_kernel: int = 5,
    dilate_iterations: int = 2,
    motion_pad_frac: float = 0.28,
    min_motion_area_ratio: float = 0.012,
    min_side_frac: float = 0.42,
    target_wh: tuple[int, int] | None = None,
    class_names: Optional[list[str]] = None,
) -> dict:

    inp = Path(input_root)
    outp = Path(output_root)
    tw, th = target_wh if target_wh is not None else (FRAME_WIDTH, FRAME_HEIGHT)
    labels = list(class_names) if class_names is not None else list(CLASS_NAMES)

    stats = {"clips": 0, "frames_ok": 0, "frames_fail": 0}

    os.chdir(_ROOT)

    for class_name in labels:
        class_in = inp / class_name
        if not class_in.is_dir():
            print(f"Skip missing class folder: {class_in}")
            continue

        for video_dir in tqdm(
            sorted(p for p in class_in.iterdir() if p.is_dir()),
            desc=f"ROI {class_name}",
            leave=False,
        ):
            stats["clips"] += 1
            rel = video_dir.relative_to(class_in)
            out_video = outp / class_name / rel

            frames = _list_frames(video_dir)
            if not frames:
                continue

            o, f = refine_video_folder(
                frames,
                out_video,
                diff_threshold=diff_threshold,
                blur_ksize=blur_ksize,
                morph_kernel=morph_kernel,
                dilate_iterations=dilate_iterations,
                motion_pad_frac=motion_pad_frac,
                min_motion_area_ratio=min_motion_area_ratio,
                min_side_frac=min_side_frac,
                target_wh=(tw, th),
            )
            stats["frames_ok"] += o
            stats["frames_fail"] += f

    print("ROI crop pipeline done:", stats)
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Motion-based ROI refinement on human crop folders."
    )
    p.add_argument(
        "--input-root",
        default=HUMAN_CROPS_DIR,
        help=f"Typically project/{HUMAN_CROPS_DIR}.",
    )
    p.add_argument(
        "--out-root",
        default=ROI_CROPS_DIR,
        help=f"Default: project/{ROI_CROPS_DIR}.",
    )
    p.add_argument("--diff-threshold", type=int, default=18)
    p.add_argument("--motion-pad-frac", type=float, default=0.28)
    p.add_argument(
        "--min-motion-area-ratio",
        type=float,
        default=0.012,
        help="If fraction of nonzero motion pixels is below this, keep full crop.",
    )
    p.add_argument("--min-side-frac", type=float, default=0.42)
    return p.parse_args()


def main() -> None:
    os.chdir(_ROOT)
    args = _parse_args()
    run_roi_crop_pipeline(
        input_root=args.input_root,
        output_root=args.out_root,
        diff_threshold=args.diff_threshold,
        motion_pad_frac=args.motion_pad_frac,
        min_motion_area_ratio=args.min_motion_area_ratio,
        min_side_frac=args.min_side_frac,
    )


if __name__ == "__main__":
    main()
