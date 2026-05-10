import os
import tensorflow as tf
from model import build_model

def convert_saved_model_to_keras():
    sm_path = "models/fight_detection_saved_model"
    keras_path = "models/fight_detection.keras"
    
    if not os.path.exists(sm_path):
        print(f"Error: {sm_path} not found. Did you finish training?")
        return

    print(f"Attempting low-level weight conversion from {sm_path}...")
    
    try:
        # 1. Rebuild the exact model architecture
        model, _ = build_model()
        
        # 2. Extract weights using TensorFlow's low-level SavedModel loader
        # This bypasses the Keras 3 'load_weights' extension check
        imported = tf.saved_model.load(sm_path)
        
        # Get all variables from the imported SavedModel
        source_vars = imported.variables
        target_vars = model.variables
        
        if len(source_vars) != len(target_vars):
            print(f"Warning: Variable count mismatch (Source: {len(source_vars)}, Target: {len(target_vars)})")
        
        # Map variables by name/order and assign
        # Note: This assumes identical architecture (which build_model provides)
        for s_var, t_var in zip(source_vars, target_vars):
            t_var.assign(s_var.numpy())
        
        # 3. Save as the new native Keras 3 format
        model.save(keras_path)
        
        print(f"Successfully converted to {keras_path}!")
        print("You can now run: python src/evaluate.py")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        print("\nAlternative: Since Keras 3 is being very strict, the most reliable path is to")
        print("run one final epoch of training with the updated src/train.py, which now")
        print("saves the correct format automatically.")

if __name__ == "__main__":
    convert_saved_model_to_keras()
