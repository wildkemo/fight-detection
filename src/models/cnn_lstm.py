"""
This file defines the model:

Input: 16 frames
CNN: extracts spatial features from each frame
LSTM: learns temporal motion
Sigmoid: outputs Violence / NonViolence

"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    LSTM,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam

from src.utils.config import (
    SEQUENCE_LENGTH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    CHANNELS,
    LEARNING_RATE,
    LSTM_UNITS
)


def build_cnn_lstm_model():
    """
    CNN + LSTM model for binary video classification.

    Input:
        sequence of frames:
        (SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)

    Output:
        probability between 0 and 1
        0 -> NonViolence
        1 -> Violence
    """

    model = Sequential()

    # ===============================
    # CNN FEATURE EXTRACTOR
    # Applied to EACH frame separately
    # ===============================

    model.add(TimeDistributed(
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        input_shape=(SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
    ))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(
        Conv2D(64, (3, 3), activation="relu", padding="same")
    ))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(
        Conv2D(128, (3, 3), activation="relu", padding="same")
    ))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Flatten()))

    # ===============================
    # TEMPORAL MODELING
    # LSTM reads the sequence of frame features
    # ===============================

    model.add(LSTM(LSTM_UNITS, return_sequences=False))

    # ===============================
    # CLASSIFIER
    # ===============================

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation="sigmoid"))

    # ===============================
    # COMPILE
    # ===============================

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_cnn_lstm_model()
    model.summary()