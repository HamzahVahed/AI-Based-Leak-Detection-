# config.py
from pathlib import Path

# === Paths ===
# Root that contains per-sensor folders; inside each sensor there are class folders.
# Example:
#   DATA_ROOT/
#     Accelerometer/Looped/<ClassName>/*.png
#     Dynamic Pressure Sensor/Looped/<ClassName>/*.png
#     Hydrophones/Looped/<ClassName>/*.png
DATA_ROOT = Path(r"C:\Users\hamza\OneDrive\Desktop\Final Design Files\cwt_cnn_lstm_project\cwt_cnn_lstm_project\input\cwt_log_new")

SENSORS = [
    "Accelerometer",
    "Dynamic Pressure Sensor",
    "Hydrophones",
]

# If your class folders are one level deeper under each sensor:
SENSOR_MODE_SUBDIR = "Looped"   # change if your images are directly under the sensor

# === Classes ===
CLASSES = [
    "No-leak",
    "Orifice Leak",
    "Gasket Leak",
    "Longitudinal Crack",
    "Circumferential Crack",
]

# === Sequence slicing (turn a spectrogram into a T-length sequence of crops) ===
SEQ_LEN = 48          # temporal slices along width
IMG_SEG_SIZE = 64     # each slice resized to (HxW) for CNN

# === Training ===
BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-3             # OneCycle peak LR
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
LSTM_HIDDEN = 384
LSTM_BIDIR = True
SEED = 42

# === Model / loss / optim extras ===
FEAT_DIM = 512
LABEL_SMOOTH = 0.03
USE_ATTENTION_POOL = True
USE_AMP = True          # amp on GPU (ignored on CPU)
MAX_GRAD_NORM = 1.0     # gradient clipping

# === SpecAugment (train only) — gentle ===
AUG_ENABLE = True
TIME_MASKS = 1
TIME_MASK_WIDTH = 0.05
FREQ_MASKS = 1
FREQ_MASK_HEIGHT = 0.05

# === Splits ===
TRAIN_PCT = 0.70
VAL_PCT   = 0.15
TEST_PCT  = 0.15

# === Saving ===
CKPT_PATH = "cnn_lstm_cwt.pt"

# === TTA (eval) ===
TTA_SHIFTS = (0, 2, 4, 6, 8)  # circular width shifts for test-time augmentation
