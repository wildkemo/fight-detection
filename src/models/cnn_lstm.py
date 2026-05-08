"""
MobileNetV3 (time-distributed) + LSTM for short video / clip classification.

Backbone: ImageNet MobileNetV3 with global pooling per frame (lightweight vs. a full
custom CNN stack, strong transfer learning). No recurrent dropout on LSTM so cuDNN
can accelerate training and inference when a GPU is available.

Input batches are expected as float32 RGB with per-pixel values either in [0, 255]
or [0, 1]; frames are normalized with ``mobilenet_v3.preprocess_input``.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import metrics
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from src.utils.config import (
    CHANNELS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LEARNING_RATE,
    LSTM_UNITS,
    SEQUENCE_LENGTH,
)


@tf.keras.utils.register_keras_serializable(package="fight_detection")
def _preprocess_frames_mobilenet_v3(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    m = tf.reduce_max(x)
    x = tf.cond(tf.less_equal(m, 1.0), lambda: x * 255.0, lambda: x)
    return preprocess_input(x)


def _spatial_backbone(name: str) -> Model:
    input_shape = (FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
    name_l = name.strip().lower()
    if name_l in ("small", "mobilenet_v3_small", "v3_small"):
        base = MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=input_shape,
            minimalistic=False,
        )
    else:
        base = MobileNetV3Large(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=input_shape,
            minimalistic=False,
        )
    base.trainable = True
    return base


def build_cnn_lstm_model(
    *,
    mobilenet_variant: str = "large",
) -> Model:
    """
    MobileNetV3 feature extractor (shared across time) + stacked LSTM + dense head.

    Parameters
    ----------
    mobilenet_variant:
        ``\"large\"`` (default) trades a bit of speed for richer features;
        ``\"small\"`` for lower latency on CPU / weak GPUs.
    """
    inp = layers.Input(
        shape=(SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS),
        name="video_clip",
    )
    x = layers.Lambda(
        _preprocess_frames_mobilenet_v3,
        name="mobilenet_v3_preprocess",
    )(inp)

    backbone = _spatial_backbone(mobilenet_variant)
    x = layers.TimeDistributed(backbone, name="time_distributed_mobilenet")(x)

    # Stacked LSTM — no recurrent_dropout so NVIDIA LSTM kernels stay enabled.
    x = layers.LSTM(
        max(LSTM_UNITS, 96),
        return_sequences=True,
        name="lstm_1",
    )(x)
    x = layers.Dropout(0.25)(x)
    x = layers.LSTM(
        max(LSTM_UNITS // 2, 48),
        return_sequences=False,
        name="lstm_2",
    )(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(1, activation="sigmoid", name="violence_prob")(x)

    model = Model(inp, out, name=f"mobilenetv3_lstm_{mobilenet_variant}")

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.AUC(name="auc"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )

    return model


if __name__ == "__main__":
    m = build_cnn_lstm_model()
    m.summary()
