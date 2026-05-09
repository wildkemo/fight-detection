"""Webcam or video-file demo for the MobileNetV3 + LSTM violence clip model."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from src.models import cnn_lstm  # noqa: F401 - registers Lambda preprocess for load_model
from src.utils.config import (
    CLASS_NAMES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MODELS_DIR,
    PREDICTION_THRESHOLD,
    SEQUENCE_LENGTH,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _load_model(which: str) -> Any:
    name = (
        "cnn_lstm_best_model.keras"
        if which == "best"
        else "cnn_lstm_final_model.keras"
    )
    path = _resolve(MODELS_DIR) / name
    if not path.is_file():
        sys.exit(f"Model not found: {path}\nTrain first: python -m src.training.train")
    try:
        return tf.keras.models.load_model(str(path), safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(str(path))


def _resize_fullframe_rgb(bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(
        bgr, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA
    )
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


_WIN_TITLE = "Fight detection - demo"


def _configure_capture(cap: Any) -> None:
    """Ask the driver for a normal preview size; helps avoid 0x0 or stuck black frames."""
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    w, h = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


def _warmup_capture(cap: Any, *, discard: int = 30) -> Optional[np.ndarray]:
    last: Optional[np.ndarray] = None
    for _ in range(discard):
        ok, frame = cap.read()
        if ok and frame is not None:
            last = frame
    return last


def _frame_looks_live(bgr: Optional[np.ndarray]) -> bool:
    if bgr is None or bgr.size == 0:
        return False
    if bgr.shape[0] < 32 or bgr.shape[1] < 32:
        return False
    m = float(np.mean(bgr))
    sd = float(np.std(bgr))
    # All-black or uninitialized buffers are ~0 mean and ~0 std
    return m > 4.0 or sd > 4.0


def _try_open_camera_index(
    idx: int,
    api: Optional[int],
) -> Tuple[Optional[Any], str]:
    cap = (
        cv2.VideoCapture(idx, api) if api is not None else cv2.VideoCapture(idx)
    )
    if not cap.isOpened():
        cap.release()
        return None, ""

    _configure_capture(cap)
    frame = _warmup_capture(cap)
    if not _frame_looks_live(frame):
        cap.release()
        return None, ""

    if api is not None and hasattr(cv2, "CAP_V4L2") and api == cv2.CAP_V4L2:
        name = "V4L2"
    else:
        name = "default"
    return cap, name


def list_camera_devices(max_index: int = 8) -> None:
    """Print which / how many video indices return a real picture (for picking --camera)."""
    print(f"Probing camera indices 0..{max_index - 1} (may take a few seconds)...\n")
    apis: list[Optional[int]] = [None]
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        apis.append(cv2.CAP_V4L2)

    any_ok = False
    for idx in range(max_index):
        for api in apis:
            cap, backend = _try_open_camera_index(idx, api)
            if cap is None:
                continue
            ok, test = cap.read()
            cap.release()
            if not (ok and test is not None):
                continue
            mh, mw = test.shape[:2]
            brightness = float(np.mean(test))
            print(
                f"  OK  index={idx}  backend={backend:8}  "
                f"frame={mw}x{mh}  mean_pixel={brightness:.1f}"
            )
            any_ok = True
            break
    if not any_ok:
        print("  (no device passed the live-frame check - check permissions / cables)")
    print("\nUse: python -m src.demo.realtime --camera <index>")


def _open_laptop_webcam(preferred_index: int, *, auto_fallback: bool) -> Any:
    """
    Open the built-in / USB webcam. Try default and V4L2 backends; reject black buffers.
    """
    apis: list[Optional[int]]
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        # Default backend first: on some systems V4L2 opens a node that exists but stays black.
        apis = [None, cv2.CAP_V4L2]
    else:
        apis = [None]

    if auto_fallback:
        if preferred_index == 0:
            order = [0, 1, 2, 3]
        else:
            order = [preferred_index, 0, 1, 2, 3]
        seen: set[int] = set()
        indices: list[int] = []
        for i in order:
            if i not in seen:
                seen.add(i)
                indices.append(i)
    else:
        indices = [preferred_index]

    for idx in indices:
        for api in apis:
            cap, backend = _try_open_camera_index(idx, api)
            if cap is not None:
                print(
                    f"Using camera index {idx} ({backend} backend). "
                    f"If the picture is wrong, run: python -m src.demo.realtime --list-cameras"
                )
                return cap

    sys.exit(
        "Could not open a working laptop/webcam (all candidates looked blank or failed).\n"
        "  - Allow camera access for this app / terminal.\n"
        "  - Close Zoom, browsers, or other apps using the camera.\n"
        "  - List devices: python -m src.demo.realtime --list-cameras\n"
        "  - Then: python -m src.demo.realtime --camera <index>"
    )


def run(
    *,
    source: Optional[str],
    camera: int,
    model_which: str,
    use_yolo: bool,
    yolo_weights: str,
    yolo_device: str,
    predict_stride: int,
    auto_camera_fallback: bool,
) -> None:
    model = _load_model(model_which)

    yolo = None
    detect_and_crop_frame = None
    if use_yolo:
        from src.data.human_detection import (
            YOLO,
            _require_ultralytics,
            detect_and_crop_frame as _detect_crop,
        )

        _require_ultralytics()
        yolo = YOLO(yolo_weights)
        detect_and_crop_frame = _detect_crop

    buf: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    last_prob = 0.0
    frame_i = 0

    if source:
        cap = cv2.VideoCapture(str(_resolve(source)))
        if not cap.isOpened():
            sys.exit(f"Cannot open video: {source}")
    else:
        print("Input: built-in / laptop webcam")
        cap = _open_laptop_webcam(camera, auto_fallback=auto_camera_fallback)

    t0 = time.perf_counter()
    shown = 0

    mode = "YOLO crop" if use_yolo else "resize full frame (faster; may differ from ROI training)"
    print(mode)
    print(f"Buffer: {SEQUENCE_LENGTH} frames @ {FRAME_WIDTH}x{FRAME_HEIGHT} | q / ESC = quit")

    try:
        cv2.namedWindow(_WIN_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WIN_TITLE, 960, 540)
    except cv2.error as exc:
        print(
            "Could not create OpenCV window (need a desktop / DISPLAY, or Qt bindings). "
            f"Detail: {exc}"
        )
        cap.release()
        sys.exit(1)

    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break

        shown += 1
        if use_yolo and yolo is not None and detect_and_crop_frame is not None:
            crop_bgr = detect_and_crop_frame(
                frame_bgr,
                yolo,
                conf=0.25,
                pad_frac=0.15,
                target_wh=(FRAME_WIDTH, FRAME_HEIGHT),
                fallback_mode="resize_full_frame",
                device=yolo_device,
            )
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = _resize_fullframe_rgb(frame_bgr)

        buf.append(rgb.astype(np.uint8))

        if len(buf) == SEQUENCE_LENGTH and frame_i % predict_stride == 0:
            clip = np.stack(list(buf), axis=0).astype(np.uint8)[np.newaxis, ...]
            last_prob = float(model.predict(clip, verbose=0)[0, 0])

        frame_i += 1

        disp = frame_bgr
        h, w = disp.shape[:2]

        if len(buf) < SEQUENCE_LENGTH:
            hud = f"Fill buffer {len(buf)}/{SEQUENCE_LENGTH}"
            cv2.putText(
                disp,
                hud,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 255),
                2,
            )
        else:
            cls_hi = CLASS_NAMES[1]
            label = f"P({cls_hi}) = {last_prob:.2f}  (thr {PREDICTION_THRESHOLD:.2f})"
            color = (
                (40, 40, 255)
                if last_prob >= PREDICTION_THRESHOLD
                else (80, 220, 80)
            )
            cv2.putText(
                disp,
                label,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            if last_prob >= PREDICTION_THRESHOLD:
                cv2.putText(
                    disp,
                    "ALERT",
                    (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

        elapsed = time.perf_counter() - t0
        fps = shown / elapsed if elapsed > 0 else 0.0
        cv2.putText(
            disp,
            f"~{fps:.1f} fps",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.imshow(_WIN_TITLE, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Real-time demo: webcam or video file + MobileNetV3 + LSTM clip model.",
    )
    p.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for the laptop webcam (default: 0).",
    )
    p.add_argument(
        "--video",
        default=None,
        help="Path to a video file (project-relative or absolute).",
    )
    p.add_argument("--model", choices=("best", "final"), default="best")
    p.add_argument(
        "--use-yolo",
        action="store_true",
        help="Person crop per frame (closer to training data; needs ultralytics).",
    )
    p.add_argument("--yolo-weights", default="yolov8n.pt")
    p.add_argument("--yolo-device", default="cpu", help="e.g. cpu or 0")
    p.add_argument(
        "--predict-every",
        type=int,
        default=1,
        metavar="N",
        help="Run the neural net every N frames (after buffer is full).",
    )
    p.add_argument(
        "--no-auto-camera",
        action="store_true",
        help="Use only --camera index (no V4L2 / other-index fallback).",
    )
    p.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe video devices and exit (use the printed index with --camera).",
    )
    args = p.parse_args()

    if args.list_cameras:
        list_camera_devices()
        return

    run(
        source=args.video,
        camera=args.camera,
        model_which=args.model,
        use_yolo=args.use_yolo,
        yolo_weights=args.yolo_weights,
        yolo_device=args.yolo_device,
        predict_stride=max(1, args.predict_every),
        auto_camera_fallback=not args.no_auto_camera,
    )


if __name__ == "__main__":
    main()
