"""
Its job:

load trained model
test it on unseen videos
calculate accuracy, precision, recall, F1
show confusion matrix

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.utils.config import (
    MODELS_DIR,
    REPORTS_DIR,
    CLASS_NAMES,
    PREDICTION_THRESHOLD
)


def evaluate():
    model_path = os.path.join(MODELS_DIR, "cnn_lstm_best_model.keras")
    test_data_path = os.path.join(REPORTS_DIR, "test_data.npz")

    print("Loading model...")
    model = load_model(model_path)

    print("Loading test data...")
    data = np.load(test_data_path)

    X_test = data["X_test"]
    y_test = data["y_test"]

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nMaking predictions...")
    y_prob = model.predict(X_test)

    y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int).reshape(-1)

    print("\nEvaluation Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()

    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()

    print(f"\nConfusion matrix saved at: {cm_path}")


if __name__ == "__main__":
    evaluate()