import tensorflow as tf
import numpy as np
import os
from pathlib import Path

# Configuration
KERAS_MODEL_PATH = "output/models/tcn_model.keras"
TFLITE_MODEL_PATH = "output/models/tcn_model_quant.tflite"
DATASET_PATH = "output/dataset/X_val_yolo.npy"

def representative_dataset_gen():
    # Load a subset of validation data for calibration
    # Using mmap_mode to save memory
    data = np.load(DATASET_PATH, mmap_mode='r')
    # Take 200 representative samples
    num_samples = min(200, len(data))
    indices = np.random.choice(len(data), num_samples, replace=False)
    
    for i in indices:
        # TFLite expects float32 for representative dataset even if quantizing to int8
        sample = data[i].astype(np.float32)
        # Expand dims to (1, 36, 51) as TFLite expects batch dimension
        yield [np.expand_dims(sample, axis=0)]

def main():
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"Error: Keras model not found at {KERAS_MODEL_PATH}")
        return

    print(f"Loading Keras model from {KERAS_MODEL_PATH}...")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    print("Converting to TFLite with INT8 quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Ensure full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()

    # Create output directory if it doesn't exist
    Path(TFLITE_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving quantized model to {TFLITE_MODEL_PATH}...")
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

    print("Success! INT8 Quantized model is ready.")

if __name__ == "__main__":
    main()
