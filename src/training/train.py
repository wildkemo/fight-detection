"""
Its job:

load sequences
build model
train model
save model
save training history

"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.data.sequence_builder import build_dataset
from src.models.cnn_lstm import build_cnn_lstm_model
from src.utils.config import (
    MODELS_DIR,
    REPORTS_DIR,
    BATCH_SIZE,
    EPOCHS,
    SEED,
    TEST_SPLIT,
    VAL_SPLIT
)


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Building dataset...")
    X, y = build_dataset()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # First split: train + temp
    temp_split = TEST_SPLIT + VAL_SPLIT

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_split,
        random_state=SEED,
        stratify=y
    )

    # Second split: validation + test
    test_ratio = TEST_SPLIT / temp_split

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_ratio,
        random_state=SEED,
        stratify=y_temp
    )

    print("\nDataset split:")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    print("\nBuilding CNN + LSTM model...")
    model = build_cnn_lstm_model()
    model.summary()

    model_path = os.path.join(MODELS_DIR, "cnn_lstm_best_model.keras")

    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("\nStarting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    final_model_path = os.path.join(MODELS_DIR, "cnn_lstm_final_model.keras")
    model.save(final_model_path)

    history_path = os.path.join(REPORTS_DIR, "training_history.npy")
    np.save(history_path, history.history)

    test_data_path = os.path.join(REPORTS_DIR, "test_data.npz")
    np.savez(
        test_data_path,
        X_test=X_test,
        y_test=y_test
    )

    print("\nTraining finished.")
    print(f"Best model saved at: {model_path}")
    print(f"Final model saved at: {final_model_path}")
    print(f"Training history saved at: {history_path}")
    print(f"Test data saved at: {test_data_path}")


if __name__ == "__main__":
    train()