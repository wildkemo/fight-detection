import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3), num_classes=1, learning_rate=1e-3):
    """
    Builds the EfficientNetB0 model with a custom classifier head.
    """
    # Base model with pretrained ImageNet weights
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model for Phase 1
    base_model.trainable = False
    
    # Define Data Augmentation layers
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ], name="data_augmentation")

    # Construct the final model
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    # Final output layer (Binary Classification)
    # No activation because we use BCEWithLogitsLoss (from_logits=True)
    outputs = layers.Dense(num_classes)(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model, base_model
