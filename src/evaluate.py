import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def evaluate_model(model_path="models/fight_detection.keras", test_dir="data_splits/test", batch_size=32):
    """
    Evaluates the trained model on the test set and prints detailed metrics.
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading test data from {test_dir}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False  # Important for confusion matrix
    )
    
    # 1. Run evaluation
    print("\n--- Standard Evaluation ---")
    results = model.evaluate(test_ds)
    # Note: model.metrics_names contains the names of the metrics
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")
        
    # 2. Detailed Metrics (Confusion Matrix)
    print("\n--- Detailed Classification Report ---")
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)
        
    y_true = np.array(y_true).flatten()
    y_pred_probs = np.array(y_pred_probs).flatten()
    
    # Convert logits (if model uses logits) or probabilities to binary classes
    # Our model uses from_logits=True, so we check if result > 0
    y_pred = (y_pred_probs > 0).astype(int)
    
    print(classification_report(y_true, y_pred, target_names=["NonViolence", "Violence"]))
    
    # 3. Confusion Matrix Plotting
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NonViolence", "Violence"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: Fight Detection")
    
    cm_path = "models/confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"\nConfusion Matrix saved to {cm_path}")

if __name__ == "__main__":
    evaluate_model()
