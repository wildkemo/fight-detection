"""
This file stores all important settings:

DATASET_DIR
OUTPUT_DIR
FRAME_SIZE
SEQUENCE_LENGTH
BATCH_SIZE
EPOCHS
LEARNING_RATE
CLASS_NAMES"""
# ===============================
# PROJECT PATHS
# ===============================

# Root dataset (raw videos)
DATASET_DIR = "dataset"

# Output directories
OUTPUT_DIR = "output"
FRAMES_DIR = f"{OUTPUT_DIR}/frames"
PROCESSED_FRAMES_DIR = f"{OUTPUT_DIR}/processed_frames"
SEQUENCES_DIR = f"{OUTPUT_DIR}/sequences"
MODELS_DIR = f"{OUTPUT_DIR}/models"
REPORTS_DIR = f"{OUTPUT_DIR}/reports"


# ===============================
# CLASSES
# ===============================

CLASS_NAMES = ["NonViolence", "Violence"]
NUM_CLASSES = 2


# ===============================
# DATA SETTINGS
# ===============================

# Frame settings
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
CHANNELS = 3   # RGB

# Sequence settings (VERY IMPORTANT for LSTM)
SEQUENCE_LENGTH = 16   # number of frames per video clip

# Frame sampling (optional)
FRAME_STRIDE = 2  # take 1 frame every N frames


# ===============================
# TRAINING SETTINGS
# ===============================

BATCH_SIZE = 4     # small because video is heavy
EPOCHS = 15
LEARNING_RATE = 0.0001


# ===============================
# MODEL SETTINGS
# ===============================

CNN_FEATURE_DIM = 128   # size of feature vector per frame
LSTM_UNITS = 64


# ===============================
# SPLIT SETTINGS
# ===============================

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


# ===============================
# THRESHOLD (for binary classification)
# ===============================

PREDICTION_THRESHOLD = 0.5


# ===============================
# RANDOM SEED (for reproducibility)
# ===============================

SEED = 42

# ===============================# ===============================
# PROJECT PATHS
# ===============================

# Root dataset (raw videos)
DATASET_DIR = "dataset"

# Output directories
OUTPUT_DIR = "output"
FRAMES_DIR = f"{OUTPUT_DIR}/frames"
PROCESSED_FRAMES_DIR = f"{OUTPUT_DIR}/processed_frames"
SEQUENCES_DIR = f"{OUTPUT_DIR}/sequences"
MODELS_DIR = f"{OUTPUT_DIR}/models"
REPORTS_DIR = f"{OUTPUT_DIR}/reports"


# ===============================
# CLASSES
# ===============================

CLASS_NAMES = ["NonViolence", "Violence"]
NUM_CLASSES = 2


# ===============================
# DATA SETTINGS
# ===============================

# Frame settings
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
CHANNELS = 3   # RGB

# Sequence settings (VERY IMPORTANT for LSTM)
SEQUENCE_LENGTH = 16   # number of frames per video clip

# Frame sampling (optional)
FRAME_STRIDE = 2  # take 1 frame every N frames


# ===============================
# TRAINING SETTINGS
# ===============================

BATCH_SIZE = 4     # small because video is heavy
EPOCHS = 15
LEARNING_RATE = 0.0001


# ===============================
# MODEL SETTINGS
# ===============================

CNN_FEATURE_DIM = 128   # size of feature vector per frame
LSTM_UNITS = 64


# ===============================
# SPLIT SETTINGS
# ===============================

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


# ===============================
# THRESHOLD (for binary classification)
# ===============================

PREDICTION_THRESHOLD = 0.5


# ===============================
# RANDOM SEED (for reproducibility)
# ===============================

SEED = 42
# PROJECT PATHS
# ===============================

# Root dataset (raw videos)
DATASET_DIR = "dataset"

# Output directories
OUTPUT_DIR = "output"
FRAMES_DIR = f"{OUTPUT_DIR}/frames"
PROCESSED_FRAMES_DIR = f"{OUTPUT_DIR}/processed_frames"
SEQUENCES_DIR = f"{OUTPUT_DIR}/sequences"
MODELS_DIR = f"{OUTPUT_DIR}/models"
REPORTS_DIR = f"{OUTPUT_DIR}/reports"


# ===============================
# CLASSES
# ===============================

CLASS_NAMES = ["NonViolence", "Violence"]
NUM_CLASSES = 2


# ===============================
# DATA SETTINGS
# ===============================

# Frame settings
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
CHANNELS = 3   # RGB

# Sequence settings (VERY IMPORTANT for LSTM)
SEQUENCE_LENGTH = 16   # number of frames per video clip

# Frame sampling (optional)
FRAME_STRIDE = 2  # take 1 frame every N frames


# ===============================
# TRAINING SETTINGS
# ===============================

BATCH_SIZE = 4     # small because video is heavy
EPOCHS = 15
LEARNING_RATE = 0.0001


# ===============================
# MODEL SETTINGS
# ===============================

CNN_FEATURE_DIM = 128   # size of feature vector per frame
LSTM_UNITS = 64


# ===============================
# SPLIT SETTINGS
# ===============================

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


# ===============================
# THRESHOLD (for binary classification)
# ===============================

PREDICTION_THRESHOLD = 0.5


# ===============================
# RANDOM SEED (for reproducibility)
# ===============================

SEED = 42