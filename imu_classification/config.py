"""Paths and constants for Beiyu IMU action recognition."""
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = DATASET_ROOT / "metadata"
SPLITS_DIR = METADATA_DIR / "splits"
INSTANCES_CSV = METADATA_DIR / "instances.csv"
ACTION_LABELS_CSV = METADATA_DIR / "action_labels.csv"

OUTPUT_DIR = DATASET_ROOT / "training" / "outputs" / "5node"
CACHE_DIR = DATASET_ROOT / "training" / "cache"

SAMPLE_RATE = 120.0
IMU_CHANNELS = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]

# PN3 Pro （ Sensor0X_IDY.csv ）
SENSOR_FILES = [
    "Sensor01_ID1.csv",  # Waist
    "Sensor02_ID2.csv",  # Palm
    "Sensor03_ID3.csv",  # Upper arm
    "Sensor04_ID4.csv",  # Forearm
    "Sensor06_ID6.csv",  # Back
]
N_SENSORS = len(SENSOR_FILES)
N_CHANNELS = N_SENSORS * len(IMU_CHANNELS)  # 30
CACHE_TAG = "5node"

DEFAULT_WIN_SIZE = 256
DEFAULT_WINDOW_MODE = "full"
MAX_WIN_SIZE = 320

SPLIT_FILES = {
    "xsub": SPLITS_DIR / "xsub_train_test.json",
    "xacq": SPLITS_DIR / "xacq_train_test.json",
    "xacq_conservative": SPLITS_DIR / "xacq_conservative_train_test.json",
}

XSUB_VAL_SUBJECTS = ["S003", "S010"]
XACQ_VAL_DATES = ["1029"]
