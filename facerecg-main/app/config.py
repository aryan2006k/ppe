"""Central config for the staff entry-recognition POC. All tunables live here."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_STORE_DIR = os.path.join(BASE_DIR, "data_store")
DB_PATH = os.path.join(DATA_STORE_DIR, "face_recognition.db")
SCHEMA_PATH = os.path.join(APP_DIR, "schema.sql")

SNAPSHOTS_DIR = os.path.join(DATA_STORE_DIR, "snapshots")
ENROLL_SNAPSHOTS_DIR = os.path.join(SNAPSHOTS_DIR, "enroll")
ENTRY_SNAPSHOTS_DIR = os.path.join(SNAPSHOTS_DIR, "entry")

# Matching
EMBEDDING_DIM = 512
MATCH_THRESHOLD = 0.65  # raised to prevent strangers from matching -- must score >= 0.65 to be AUTHORIZED

# Enrollment
MIN_ENROLL_IMAGES = 3  # minimum successful face detections required to enroll a person
BURST_CAPTURE_COUNT = 8
BURST_CAPTURE_INTERVAL_SEC = 1.2

# Entry monitor
CAMERA_INDEX = 0
LOG_COOLDOWN_SECONDS = 15
DETECTION_DOWNSCALE = 0.5  # resize factor applied before detection; boxes scaled back up

# Detector
DETECTOR_SCORE_THRESHOLD = 0.6  # confidence cutoff; lowered from 0.9 since
# occluded (mask/cap) faces score lower confidence than full unoccluded faces
YUNET_MODEL_PATH = os.path.join(APP_DIR, "models", "face_detection_yunet_2023mar.onnx")
YUNET_NMS_THRESHOLD = 0.3
