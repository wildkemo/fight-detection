import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "data/sequences"
MODEL_PATH = "models/tcn_model.tflite"
SEQ_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

def load_data():
    """Loads preprocessed sequence data."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    
    print(f"Loaded data: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def residual_block(x, filters, dilation_rate, dropout_rate=0.2):
    """A single residual block with dilated convolutions."""
    shortcut = x
    
    # Branch 1
    x = layers.Conv1D(filters, kernel_size=3, padding='causal', dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(filters, kernel_size=3, padding='causal', dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    
    # Match shortcut dimensions if necessary
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, padding='same')(shortcut)
    
    res = layers.Add()([shortcut, x])
    return layers.Activation('relu')(res)

def build_model(input_shape, num_classes=1):
    """Builds the Residual TCN model."""
    inputs = layers.Input(shape=input_shape)
    
    # Stack residual blocks with increasing dilation
    x = residual_block(inputs, filters=64, dilation_rate=1)
    x = residual_block(x, filters=64, dilation_rate=2)
    x = residual_block(x, filters=128, dilation_rate=4)
    x = residual_block(x, filters=128, dilation_rate=8)
    x = residual_block(x, filters=256, dilation_rate=16)
    
    # Aggregation - Use both Average and Max pooling to capture steady motion and impacts
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    
    # Final classifier
    x = layers.Dense(64, activation='relu')(x)

    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def optimize_threshold(y_true, y_probs):
    """Finds the threshold that maximizes F1-score."""
    thresholds = np.linspace(0.05, 0.95, 50)
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"Optimal Threshold: {best_thresh:.2f} (F1-score: {best_f1:.4f})")
    return best_thresh

def convert_to_tflite(keras_model, X_val, output_path):
    """Converts the model to INT8 quantized TFLite."""
    print("Converting to TFLite with INT8 quantization...")
    
    def representative_data_gen():
        for i in range(100):
            # Sample from validation data
            idx = np.random.randint(0, len(X_val))
            sample = X_val[idx:idx+1].astype(np.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Option A: Stable conversion (keep float IO)
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {output_path}")

def main():
    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Robustness: Clip NaNs
    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)
    
    # 2. Compute Class Weights
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {int(c): w for c, w in zip(np.unique(y_train), weights)}
    print(f"Class Weights: {class_weights}")
    
    # 3. Build & Train Model
    assert X_train.ndim == 3, f"Invalid shape: {X_train.shape}"
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    model.summary()
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # 4. Threshold Optimization on Validation Set
    val_probs = model.predict(X_val)
    best_thresh = optimize_threshold(y_val, val_probs)
    
    # 5. Final Evaluation on Test Set
    test_probs = model.predict(X_test)
    test_preds = (test_probs >= best_thresh).astype(int)
    print("\n--- Final Test Report ---")
    report = classification_report(y_test, test_preds, output_dict=True)
    print(classification_report(y_test, test_preds))
    
    # 6. Save Metrics to JSON
    metrics_path = "models/metrics.json"
    import json
    metrics_data = {
        "optimal_threshold": float(best_thresh),
        "test_report": report,
        "history": {k: [float(v) for v in l] for k, l in history.history.items()}
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")
    
    # 7. Export to TFLite
    convert_to_tflite(model, X_val, MODEL_PATH)

if __name__ == "__main__":
    main()
