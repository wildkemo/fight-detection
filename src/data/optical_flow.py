"""
Dense optical flow between consecutive frames.

Uses Gunnar Farnebäck’s method (OpenCV): per-pixel displacement (dx, dy).
From that we derive motion **magnitude** (intensity), **direction**, and per-frame
two-channel motion fields your sequence model can ingest as a **JPEG stack**
(one encoded image per time step).

Typical workflow (after ROI crops)::

    ROI_CROPS_DIR / <class> / <video_id> / frame_*.jpg

Output::

    OPTICAL_FLOW_DIR / <class> / <video_id> / frame_*.jpg

Frame ``t`` encodes flow from grayscale(t-1) → grayscale(t).
Frame ``0`` uses duplicated ``t`` as ``t-1`` → near-zero displacement.

Requires: OpenCV, NumPy.

Run::

    PYTHONPATH=. python src/data/optical_flow.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Literal, Optional, TypedDict

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
    OPTICAL_FLOW_DIR,
    ROI_CROPS_DIR,
)

_FRAME_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

EncodeMode = Literal["hsv", "vector_mag"]


class FlowPolar(TypedDict):
    magnitude: np.ndarray
    angle: np.ndarray


class FlowStats(TypedDict):
    mean_magnitude: float
    median_magnitude: float
    std_magnitude: float


def _list_frames(video_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in _FRAME_GLOBS:
        files.extend(video_dir.glob(pattern))
    return sorted({p.resolve(): p for p in files}.values(), key=lambda p: p.name)


def preprocess_gray(frame_bgr: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    k = max(3, blur_ksize | 1)
    return cv2.GaussianBlur(g, (k, k), 0)


def dense_flow_farneback(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    *,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
) -> np.ndarray:
    """Return flow ``(H, W, 2) float32``: channel 0 = dx, channel 1 = dy."""
    return cv2.calcOpticalFlowFarneback(
        gray_prev,
        gray_curr,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    )


def flow_to_polar(flow: np.ndarray) -> FlowPolar:
    gx = flow[..., 0].astype(np.float32)
    gy = flow[..., 1].astype(np.float32)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    return FlowPolar(magnitude=mag.astype(np.float32), angle=ang.astype(np.float32))


def flow_global_stats(flow: np.ndarray) -> FlowStats:
    pol = flow_to_polar(flow)
    m = pol["magnitude"].reshape(-1)
    return FlowStats(
        mean_magnitude=float(np.mean(m)),
        median_magnitude=float(np.median(m)),
        std_magnitude=float(np.std(m)),
    )


def flow_to_bgr_hsv_visual(
    flow: np.ndarray,
    *,
    saturation: int = 255,
    mag_percentile_clip: float = 98.0,
) -> np.ndarray:
    """Hue ~ direction (0°–360° mapped to OpenCV hue), value ~ motion intensity."""
    pol = flow_to_polar(flow)
    mag = pol["magnitude"]
    ang = pol["angle"]

    h_deg = np.mod(ang, 2.0 * math.pi) / (2.0 * math.pi)
    hue = np.clip((h_deg * 179).astype(np.float32), 0, 179).astype(np.uint8)

    cap = float(max(np.percentile(mag, mag_percentile_clip), 1e-6))
    vmag = np.clip(mag.astype(np.float32) / cap, 0.0, 1.0)
    val = (vmag * 255).astype(np.uint8)

    hsv = np.zeros((mag.shape[0], mag.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = np.uint8(saturation)
    hsv[..., 2] = val
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flow_to_bgr_vector_mag(
    flow: np.ndarray,
    *,
    mag_percentile_clip: float = 98.0,
) -> np.ndarray:
    """
    Encoder-friendly RGB-like BGR image:

    - **B**: motion magnitude scaled to `[0, 255]`.
    - **G / R**: horizontal / vertical displacement, ~128 means “no motion” on that axis.
    """
    gx = flow[..., 0].astype(np.float32)
    gy = flow[..., 1].astype(np.float32)
    pol = flow_to_polar(flow)
    mag = pol["magnitude"]

    clip_m = float(max(np.percentile(mag, mag_percentile_clip), 1e-6))
    bm = np.clip(mag / clip_m * 255.0, 0.0, 255.0).astype(np.uint8)

    stacked = np.abs(np.concatenate([gx.reshape(-1), gy.reshape(-1)]))
    vmax = float(max(np.percentile(stacked, mag_percentile_clip), 1e-6))
    gv = np.clip(gx / vmax * 127.5 + 127.5, 0, 255).astype(np.uint8)
    rv = np.clip(gy / vmax * 127.5 + 127.5, 0, 255).astype(np.uint8)
    return cv2.merge([bm, gv, rv])


def encode_flow_bgr(
    flow: np.ndarray,
    mode: EncodeMode,
    *,
    mag_percentile_clip: float = 98.0,
    saturation: int = 255,
) -> np.ndarray:
    if mode == "hsv":
        return flow_to_bgr_hsv_visual(
            flow,
            mag_percentile_clip=mag_percentile_clip,
            saturation=saturation,
        )
    if mode == "vector_mag":
        return flow_to_bgr_vector_mag(flow, mag_percentile_clip=mag_percentile_clip)
    raise ValueError(f"unknown encode mode: {mode}")


def flow_sequence_from_frames(
    frames_bgr: list[np.ndarray],
    *,
    gray_blur_ksize: int = 5,
    fb_kwargs: Optional[dict] = None,
) -> list[np.ndarray]:
    if not frames_bgr:
        return []

    fb = fb_kwargs if fb_kwargs is not None else {}
    grays = [preprocess_gray(f, gray_blur_ksize) for f in frames_bgr]

    flows: list[np.ndarray] = []
    for i, g_curr in enumerate(grays):
        gray_prev = g_curr.copy() if i == 0 else grays[i - 1]
        fw = dense_flow_farneback(gray_prev, g_curr, **fb)
        flows.append(fw.astype(np.float32))

    return flows


def process_clip_folder(
    frame_paths: list[Path],
    out_dir: Path,
    *,
    encode_mode: EncodeMode,
    resize_wh: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT),
    jpeg_quality: int = 92,
    mag_percentile_clip: float = 98.0,
) -> tuple[int, int]:
    imgs: list[np.ndarray] = []
    fails = 0
    for p in frame_paths:
        im = cv2.imread(str(p))
        if im is None:
            fails += 1
            imgs.append(np.zeros((resize_wh[1], resize_wh[0], 3), dtype=np.uint8))
            continue
        imgs.append(im)

    if not imgs:
        return 0, fails

    tw, th = resize_wh

    def resize_if_needed(x: np.ndarray) -> np.ndarray:
        if x.shape[1] == tw and x.shape[0] == th:
            return x
        return cv2.resize(x, (tw, th), interpolation=cv2.INTER_AREA)

    resized = [resize_if_needed(im) for im in imgs]
    flows = flow_sequence_from_frames(resized)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for p, fw in zip(frame_paths, flows):
        visual = encode_flow_bgr(
            fw,
            encode_mode,
            mag_percentile_clip=mag_percentile_clip,
        )
        dest = out_dir / p.name
        if dest.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            dest = dest.with_suffix(".jpg")
        cv2.imwrite(str(dest), visual, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        written += 1

    return written, fails


def run_optical_flow_pipeline(
    input_root: str | Path = ROI_CROPS_DIR,
    output_root: str | Path = OPTICAL_FLOW_DIR,
    *,
    encode_mode: EncodeMode = "vector_mag",
    resize_wh: tuple[int, int] | None = None,
    mag_percentile_clip: float = 98.0,
    class_names: Optional[list[str]] = None,
) -> dict:
    inp = Path(input_root)
    outp = Path(output_root)
    tw, th = resize_wh if resize_wh is not None else (FRAME_WIDTH, FRAME_HEIGHT)
    labels = list(class_names) if class_names is not None else list(CLASS_NAMES)

    stats = {"clips": 0, "frames_written": 0, "read_errors": 0}
    os.chdir(_ROOT)

    for class_name in labels:
        class_in = inp / class_name
        if not class_in.is_dir():
            print(f"Skip missing class folder: {class_in}")
            continue

        vids = sorted(p for p in class_in.iterdir() if p.is_dir())
        for video_dir in tqdm(vids, desc=f"Flow {class_name}", leave=False):
            stats["clips"] += 1
            rel = video_dir.relative_to(class_in)
            out_video = outp / class_name / rel

            frames = _list_frames(video_dir)
            if not frames:
                continue
            wr, fa = process_clip_folder(
                frames,
                out_video,
                encode_mode=encode_mode,
                resize_wh=(tw, th),
                mag_percentile_clip=mag_percentile_clip,
            )
            stats["frames_written"] += wr
            stats["read_errors"] += fa

    print("Optical flow pipeline:", stats)
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Farnebäck dense optical-flow JPEG stacks.")
    p.add_argument("--input-root", default=ROI_CROPS_DIR, help=f"Default project/{ROI_CROPS_DIR}")
    p.add_argument("--out-root", default=OPTICAL_FLOW_DIR, help=f"Default project/{OPTICAL_FLOW_DIR}")
    p.add_argument(
        "--encode",
        choices=("hsv", "vector_mag"),
        default="vector_mag",
        help="hsv: rainbow direction+hue | vector_mag: |v|, vx, vy in B,G,R",
    )
    return p.parse_args()


def main() -> None:
    os.chdir(_ROOT)
    args = _parse_args()
    run_optical_flow_pipeline(
        input_root=args.input_root,
        output_root=args.out_root,
        encode_mode=args.encode,
    )


if __name__ == "__main__":
    main()
