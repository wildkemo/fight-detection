"""
CNN + LSTM for Video Classification
Improved version: smaller, faster, less overfitting
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed,
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    LSTM,
    BatchNormalization,
    GlobalAveragePooling2D,
    Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from src.utils.config import (
    SEQUENCE_LENGTH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    CHANNELS,
    LEARNING_RATE,
    LSTM_UNITS
)


def build_cnn_lstm_model():

    model = Sequential([
        # ✅ Proper input layer (fix warning)
        Input(shape=(SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)),

        # ===============================
        # CNN FEATURE EXTRACTOR
        # ===============================

        TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same',
                              kernel_regularizer=l2(0.001))),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),

        TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='same',
                              kernel_regularizer=l2(0.001))),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),

        TimeDistributed(Conv2D(128, (3,3), activation='relu', padding='same',
                              kernel_regularizer=l2(0.001))),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),

        # ✅ KEY FIX: reduce features drastically
        TimeDistributed(GlobalAveragePooling2D()),

        # ===============================
        # TEMPORAL MODELING
        # ===============================

        LSTM(LSTM_UNITS, dropout=0.3, recurrent_dropout=0.3),

        # ===============================
        # CLASSIFIER
        # ===============================

        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.5),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_cnn_lstm_model()
    model.summary()
