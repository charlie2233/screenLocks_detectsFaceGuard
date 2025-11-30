import sys
from pathlib import Path

APP_NAME = "FaceLock Guardian v2"

# Base paths
APPDATA = Path.home() / "AppData" / "Roaming"
DATA_DIR = APPDATA / "FaceLockGuardian"
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Files
EMBEDDING_PATH = DATA_DIR / "embedding.bin"
CONFIG_PATH = DATA_DIR / "config.json"
PASSWORD_PATH = DATA_DIR / "password.bin"
KEY_PATH = DATA_DIR / "key.bin"

# Defaults
DEFAULT_LOCK_DELAY = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.45
DEFAULT_CAMERA_INDEX = 0
CAMERA_INDEX = 1
DEFAULT_STATIC_VARIANCE_THRESHOLD = 5.0
MOTION_MIN_DRIFT = 0.0015
EAR_THRESHOLD = 0.2
EAR_CONSEC_FRAMES = 2
UNLOCK_GRACE_PERIOD = 300  # seconds

YOLO_MODEL_NAME = "yolov8n-face.pt"
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt"
FACE_EMBEDDING_MODEL_URL = "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx"
FACE_EMBEDDING_MODEL_MIRRORS = [
    "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/resolve/main/arcfaceresnet100-11-int8.onnx",
    "https://github.com/xiangyuecn/ArcFaceONNX/raw/master/models/arcface_r100.onnx",
]
FACE_EMBEDDING_MODEL_PATH = MODELS_DIR / "face_embedding.onnx"

REGISTRY_RUN_KEY = r"Software\\Microsoft\\Windows\\CurrentVersion\\Run"


def executable_path() -> Path:
    """Return current executable path (PyInstaller safe)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable)
    return Path(__file__).resolve().parent.parent / "main.py"
