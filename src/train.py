import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
from pathlib import Path

# Configuration
DATASET_DIR = Path("output/dataset")
MODEL_DIR = Path("output/models")
EVAL_DIR = Path("output/eval")
MODEL_PATH = MODEL_DIR / "tcn_model.keras"
METRICS_PATH = EVAL_DIR / "tcn_metrics.json"

def main():
    print("\n" + "="*60)
    print("!!! FIGHT DETECTION SYSTEM - TCN MODEL TRAINING !!!")
    
    # --- GPU CONFIGURATION ---
    # Explicitly check for GPU and enable memory growth to prevent VRAM hoarding
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[+] TensorFlow GPU DETECTED: {len(gpus)} GPU(s) found.")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"    - Enabled VRAM memory growth for: {gpu.name}")
            except RuntimeError as e:
                print(f"    - Error setting memory growth: {e}")
    else:
        print("[!] WARNING: TensorFlow did NOT detect a GPU. Training will run on slow CPU.")
    print("="*60 + "\n")

    print("Loading datasets (YOLO-Pose versions)...")
    try:
        X_train = np.load(DATASET_DIR / "X_train_yolo.npy")
        y_train = np.load(DATASET_DIR / "y_train_yolo.npy")
        X_val = np.load(DATASET_DIR / "X_val_yolo.npy")
        y_val = np.load(DATASET_DIR / "y_val_yolo.npy")
        X_test = np.load(DATASET_DIR / "X_test_yolo.npy")
        y_test = np.load(DATASET_DIR / "y_test_yolo.npy")
    except FileNotFoundError:
        print(f"Error: Dataset files not found in {DATASET_DIR}. Please run build_yolo_sequences first.")
        return

    print(f"[*] Train samples: {len(X_train)}")
    print(f"[*] Val samples: {len(X_val)}")
    print(f"[*] Test samples: {len(X_test)}")
    print(f"[*] Sequence Shape: {X_train.shape[1:]} (Steps, Features)")

    # Step 1: Compute Class Weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    # Extreme focus on Recall (Zero Misses Goal)
    class_weights[1] = class_weights[1] * 3.0
    print(f"[*] Adjusted Class weights for high recall: {class_weights}")

    # Step 1.5: Build Optimized tf.data Pipeline
    BATCH_SIZE = 1024
    print(f"[*] Building high-performance data pipeline (Batch Size: {BATCH_SIZE})...")
    
    def create_dataset(X, y, is_training=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if is_training:
            ds = ds.shuffle(buffer_size=len(X))
        ds = ds.cache() # Keep data in VRAM/RAM after first epoch
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE) # Prepare next batch while GPU trains
        return ds

    train_ds = create_dataset(X_train, y_train, is_training=True)
    val_ds = create_dataset(X_val, y_val)
    test_ds = create_dataset(X_test, y_test)

    # Step 2: Build Model (TCN / 1D-CNN Architecture)
    # This architecture is superior for coordinate-based motion analysis
    model = Sequential([
        # Block 1: Local temporal patterns
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal', 
               input_shape=(X_train.shape[1], X_train.shape[2])),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Block 2: Broader temporal patterns
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Global temporal aggregation
        GlobalAveragePooling1D(),
        
        # Classifier
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Step 3: Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    # Step 4: Training
    print("\nStarting training...")
    epochs = 100 
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Step 5: Save Model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n[+] Model saved to {MODEL_PATH}")

    # Step 6: TCN-Only Evaluation
    print("Evaluating model on test set...")
    y_pred_prob = model.predict(test_ds)
    # Low threshold for high recall goal
    y_pred = (y_pred_prob > 0.3).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "summary": report["weighted avg"],
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "details": report,
        "architecture": "TCN (1D-CNN)",
        "input_shape": list(X_train.shape[1:])
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"[+] Metrics saved to {METRICS_PATH}")
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
