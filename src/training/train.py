"""Load fixed-length RGB clips from ``output/`` and train ``src.models.cnn_lstm``."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.models import cnn_lstm
from src.utils.config import (
    BATCH_SIZE,
    CHANNELS,
    CLASS_NAMES,
    EPOCHS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MODELS_DIR,
    REPORTS_DIR,
    ROI_CROPS_DIR,
    SEED,
    SEQUENCE_LENGTH,
    TEST_SPLIT,
    VAL_SPLIT,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Frame tree: ``<this>/<NonViolence|Violence>/<video_id>/frame_*.jpg``
# (same layout as ``crop_roi`` / ``human_detection`` under ``output/``).
TRAINING_DATA_ROOT = ROI_CROPS_DIR

MOBILENET_VARIANT = "large"

_FRAME_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def _resolve_under_project(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _list_frames(video_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in _FRAME_GLOBS:
        files.extend(video_dir.glob(pattern))
    return sorted({q.resolve(): q for q in files}.values(), key=lambda q: q.name)


def _load_clip(paths: list[Path], *, n_frames: int, h: int, w: int) -> np.ndarray:
    frames: list[np.ndarray] = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(rgb)

    if not frames:
        return np.zeros((n_frames, h, w, 3), dtype=np.uint8)

    if len(frames) < n_frames:
        last = frames[-1]
        while len(frames) < n_frames:
            frames.append(last.copy())
    elif len(frames) > n_frames:
        idx = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        frames = [frames[int(i)] for i in idx]

    return np.stack(frames, axis=0).astype(np.uint8)


def load_xy_from_output(frames_root: Path) -> tuple[np.ndarray, np.ndarray]:
    clips: list[np.ndarray] = []
    labels: list[int] = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = frames_root / class_name
        if not class_dir.is_dir():
            continue
        for video_dir in sorted(d for d in class_dir.iterdir() if d.is_dir()):
            paths = _list_frames(video_dir)
            if not paths:
                continue
            clip = _load_clip(
                paths,
                n_frames=SEQUENCE_LENGTH,
                h=FRAME_HEIGHT,
                w=FRAME_WIDTH,
            )
            clips.append(clip)
            labels.append(label)

    if not clips:
        raise FileNotFoundError(
            f"No clips under {frames_root} "
            f"(expected subfolders {list(CLASS_NAMES)} with video dirs and frames). "
            "Run preprocessing (e.g. ``python main.py``) so ``output/`` is populated."
        )

    X = np.stack(clips, axis=0)
    y = np.asarray(labels, dtype=np.float32)
    return X, y


def train() -> None:
    models_dir = _resolve_under_project(MODELS_DIR)
    reports_dir = _resolve_under_project(REPORTS_DIR)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    data_root = _resolve_under_project(TRAINING_DATA_ROOT)
    print(f"Loading clips from: {data_root}")
    X, y = load_xy_from_output(data_root)
    print("X shape:", X.shape, "y shape:", y.shape)

    expected = (SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
    if X.ndim != 5 or tuple(X.shape[1:]) != expected:
        raise ValueError(
            f"Expected X shape (N, {', '.join(map(str, expected))}), got {X.shape}"
        )

    temp_split = TEST_SPLIT + VAL_SPLIT
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_split,
        random_state=SEED,
        stratify=y,
    )
    test_ratio = TEST_SPLIT / temp_split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_ratio,
        random_state=SEED,
        stratify=y_temp,
    )

    model = cnn_lstm.build_cnn_lstm_model(mobilenet_variant=MOBILENET_VARIANT)
    model.summary()

    model_path = os.path.join(models_dir, "cnn_lstm_best_model.keras")
    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    final_model_path = os.path.join(models_dir, "cnn_lstm_final_model.keras")
    model.save(final_model_path)
    np.save(os.path.join(reports_dir, "training_history.npy"), history.history)
    np.savez(
        os.path.join(reports_dir, "test_data.npz"),
        X_test=X_test,
        y_test=y_test,
    )

    print(
        f"Training finished.\nBest: {model_path}\nFinal: {final_model_path}\n"
        f"History: {reports_dir}/training_history.npy\n"
        f"Test split: {reports_dir}/test_data.npz"
    )


if __name__ == "__main__":
    train()
