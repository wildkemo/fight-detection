import os
import tensorflow as tf
from pathlib import Path
from model import build_model

def train_model(data_dir="data_splits", batch_size=32, epochs_phase1=5, epochs_phase2=10):
    """
    Orchestrates the two-phase training process.
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    
    # 1. Load Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode='binary'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode='binary'
    )
    
    # Optimization for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # 2. Build Model (Phase 1: Warmup)
    print("\n--- Phase 1: Training Classifier Head ---")
    model, base_model = build_model(learning_rate=1e-3)
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1
    )
    
    # 3. Fine-Tuning (Phase 2)
    print("\n--- Phase 2: Fine-tuning Full Model ---")
    base_model.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase2
    )
    
    # 4. Save and Export to TFLite
    print("\n--- Exporting Model ---")
    model_save_path = "models/fight_detection_saved_model"
    os.makedirs("models", exist_ok=True)
    
    # Use export() for SavedModel format in Keras 3
    model.export(model_save_path)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
    # Default optimization (Baseline)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    tflite_path = "models/fight_detection.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"SavedModel exported to {model_save_path}")
    print(f"TFLite model saved to {tflite_path}")

if __name__ == "__main__":
    train_model()
