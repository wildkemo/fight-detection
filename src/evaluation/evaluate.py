"""Evaluate the trained MobileNetV3 + LSTM clip classifier on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.models import cnn_lstm as _cnn_lstm_entry  # noqa: F401 - registers Lambda fn for load_model

from src.utils.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    MODELS_DIR,
    PREDICTION_THRESHOLD,
    REPORTS_DIR,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_under_project(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _load_classifier(model_path: Path) -> tf.keras.Model:
    """Restore weights for ``cnn_lstm``; MobileNet Lambda preprocess is Keras-registered."""
    try:
        return tf.keras.models.load_model(str(model_path), safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(str(model_path))


def evaluate(
    *,
    model_path: Path | None = None,
    test_data_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    reports_dir = _resolve_under_project(REPORTS_DIR)
    models_dir = _resolve_under_project(MODELS_DIR)

    if model_path is None:
        model_path = models_dir / "cnn_lstm_best_model.keras"
    else:
        model_path = _resolve_under_project(model_path)

    if test_data_path is None:
        test_data_path = reports_dir / "test_data.npz"
    else:
        test_data_path = _resolve_under_project(test_data_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing model weights: {model_path}\n"
            "Train first with: python -m src.training.train"
        )
    if not test_data_path.is_file():
        raise FileNotFoundError(
            f"Missing test bundle: {test_data_path}\n"
            "Training writes this after splitting (see src/training/train.py)."
        )

    print(f"Loading model: {model_path}")
    model = _load_classifier(model_path)
    model.summary()

    print(f"Loading test data: {test_data_path}")
    data = np.load(test_data_path)
    X_test = np.asarray(data["X_test"])
    y_test = np.asarray(data["y_test"]).astype(int).reshape(-1)

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nTensorFlow metrics (compiled loss + metrics)...")
    try:
        metrics_dict = model.evaluate(
            X_test,
            y_test.astype(np.float32),
            batch_size=BATCH_SIZE,
            verbose=1,
            return_dict=True,
        )
        for name, value in metrics_dict.items():
            print(f"  {name}: {float(value):.6f}")
    except TypeError:
        ev = model.evaluate(
            X_test, y_test.astype(np.float32), batch_size=BATCH_SIZE, verbose=1
        )
        print(f"  results: {ev}")

    print("\nPredicting...")
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)

    print("\nScikit-learn (threshold = {:.2f}):".format(PREDICTION_THRESHOLD))
    print("  Accuracy:  ", accuracy_score(y_test, y_pred))
    print("  Precision: ", precision_score(y_test, y_pred, zero_division=0))
    print("  Recall:    ", recall_score(y_test, y_pred, zero_division=0))
    print("  F1-score:  ", f1_score(y_test, y_pred, zero_division=0))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix — MobileNetV3 + LSTM (test set)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()

    cm_path = reports_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"\nConfusion matrix saved to: {cm_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate MobileNetV3 + LSTM on the saved train/test split.",
    )
    p.add_argument(
        "--model",
        choices=("best", "final"),
        default="best",
        help="Which checkpoint to load (default: best val_loss).",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Override path to a .keras file (wins over --model).",
    )
    p.add_argument(
        "--test-data",
        default=None,
        help="Override path to test_data.npz.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Save confusion matrix PNG only (no GUI).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.model_path:
        mp: Path | None = Path(args.model_path)
    else:
        name = (
            "cnn_lstm_best_model.keras"
            if args.model == "best"
            else "cnn_lstm_final_model.keras"
        )
        mp = _resolve_under_project(MODELS_DIR) / name

    td = Path(args.test_data) if args.test_data else None
    evaluate(model_path=mp, test_data_path=td, show_plot=not args.no_show)


if __name__ == "__main__":
    main()
