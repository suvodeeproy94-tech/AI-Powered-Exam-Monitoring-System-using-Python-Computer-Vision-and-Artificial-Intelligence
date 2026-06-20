"""Application paths, visual constants, and user-editable settings.

The dashboard loads one AppSettings object when it starts. The settings dialog
updates that object and saves it to settings.json so the next run uses the same
camera and monitoring choices.
"""

from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path


# Project folders are created automatically on the first import.
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports" / "output"
EVIDENCE_DIR = BASE_DIR / "evidence"
DATA_DIR = BASE_DIR / "data"
GESTURE_DATA_DIR = DATA_DIR / "gesture_samples"
ASSET_DIR = BASE_DIR / "assets"
MODEL_DIR = ASSET_DIR / "models"
CACHE_DIR = BASE_DIR / ".cache"
SETTINGS_FILE = BASE_DIR / "settings.json"
ACTIVITY_LOG_FILE = LOG_DIR / "activity_log.csv"
SYSTEM_LOG_FILE = LOG_DIR / "system.log"
DATABASE_FILE = DATA_DIR / "exam_monitoring.db"
FACE_DETECTOR_MODEL = MODEL_DIR / "blaze_face_short_range.tflite"
FACE_LANDMARKER_MODEL = MODEL_DIR / "face_landmarker.task"
HAND_LANDMARKER_MODEL = MODEL_DIR / "hand_landmarker.task"
YUNET_FACE_DETECTOR_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
GESTURE_CLASSIFIER_MODEL = MODEL_DIR / "gesture_classifier.joblib"

for required_directory in (
    LOG_DIR,
    REPORT_DIR,
    EVIDENCE_DIR,
    GESTURE_DATA_DIR,
    ASSET_DIR,
    MODEL_DIR,
    CACHE_DIR,
):
    required_directory.mkdir(parents=True, exist_ok=True)

# Keep Matplotlib cache files inside the project on restricted Windows systems.
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))


@dataclass
class AppSettings:
    """Store all values that may change how monitoring works."""

    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    mirror_camera: bool = True

    face_detection_confidence: float = 0.60
    face_tracking_confidence: float = 0.60
    yunet_face_detection_enabled: bool = True
    yunet_score_threshold: float = 0.75
    yunet_nms_threshold: float = 0.30
    yunet_top_k: int = 5000
    hand_detection_confidence: float = 0.60
    hand_tracking_confidence: float = 0.55
    max_hands: int = 2
    enhance_low_light: bool = True
    low_light_threshold: float = 90.0
    blur_threshold: float = 45.0
    calibration_enabled: bool = True
    calibration_seconds: float = 5.0
    calibration_min_valid_ratio: float = 0.70
    minimum_face_size_ratio: float = 0.04
    maximum_face_size_ratio: float = 0.45

    expected_face_count: int = 1
    face_missing_frames: int = 12
    multiple_face_frames: int = 3
    face_outside_frames: int = 10
    look_away_frames: int = 15
    look_away_ratio_threshold: float = 0.18
    head_pose_enabled: bool = True
    head_yaw_threshold: float = 24.0
    head_pitch_threshold: float = 20.0
    gaze_tracking_enabled: bool = True
    gaze_horizontal_threshold: float = 0.22
    gaze_vertical_threshold: float = 0.25
    eye_closed_threshold: float = 0.18
    edge_margin_ratio: float = 0.03
    face_tracking_grace_frames: int = 3

    hand_movement_threshold: float = 40.0
    hand_movement_ratio_threshold: float = 0.28
    excessive_hand_frames: int = 8
    hand_cover_frames: int = 5
    hand_cover_min_landmarks: int = 5
    face_movement_threshold: float = 35.0
    face_movement_ratio_threshold: float = 0.18
    frequent_movement_frames: int = 8
    face_cover_overlap_ratio: float = 0.25
    suspicious_gesture_frames: int = 4
    gesture_history_frames: int = 7
    gesture_stable_frames: int = 3
    gesture_majority_ratio: float = 0.60
    gesture_min_confidence: float = 0.58
    minimum_face_area_ratio: float = 0.015
    trained_gesture_model_enabled: bool = True
    trained_gesture_min_confidence: float = 0.70

    face_missing_seconds: float = 1.5
    multiple_face_seconds: float = 0.7
    face_outside_seconds: float = 1.2
    look_away_seconds: float = 2.0
    eyes_closed_seconds: float = 3.0
    hand_cover_seconds: float = 1.0
    excessive_hand_seconds: float = 1.2
    frequent_movement_seconds: float = 1.2
    suspicious_gesture_seconds: float = 1.0

    evidence_capture_enabled: bool = True
    evidence_minimum_level: str = "WARNING"
    risk_decay_per_second: float = 1.5
    risk_warning_threshold: float = 35.0
    risk_critical_threshold: float = 70.0

    alert_cooldown_seconds: float = 5.0
    logging_enabled: bool = True
    draw_face_mesh: bool = False

    def validate(self):
        """Keep saved values inside safe and practical limits."""
        self.camera_index = max(0, int(self.camera_index))
        self.camera_width = max(320, min(1920, int(self.camera_width)))
        self.camera_height = max(240, min(1080, int(self.camera_height)))
        self.camera_fps = max(10, min(60, int(self.camera_fps)))

        self.face_detection_confidence = _clamp(self.face_detection_confidence, 0.1, 1.0)
        self.face_tracking_confidence = _clamp(self.face_tracking_confidence, 0.1, 1.0)
        self.yunet_score_threshold = _clamp(
            self.yunet_score_threshold, 0.30, 0.95
        )
        self.yunet_nms_threshold = _clamp(self.yunet_nms_threshold, 0.1, 0.9)
        self.yunet_top_k = max(100, min(10000, int(self.yunet_top_k)))
        self.hand_detection_confidence = _clamp(self.hand_detection_confidence, 0.1, 1.0)
        self.hand_tracking_confidence = _clamp(self.hand_tracking_confidence, 0.1, 1.0)
        self.max_hands = max(1, min(2, int(self.max_hands)))
        self.low_light_threshold = _clamp(self.low_light_threshold, 20.0, 180.0)
        self.blur_threshold = max(1.0, float(self.blur_threshold))
        self.calibration_seconds = _clamp(self.calibration_seconds, 1.0, 30.0)
        self.calibration_min_valid_ratio = _clamp(
            self.calibration_min_valid_ratio, 0.20, 1.0
        )
        self.minimum_face_size_ratio = _clamp(
            self.minimum_face_size_ratio, 0.01, 0.40
        )
        self.maximum_face_size_ratio = _clamp(
            self.maximum_face_size_ratio,
            self.minimum_face_size_ratio + 0.01,
            0.95,
        )

        self.expected_face_count = max(1, int(self.expected_face_count))
        self.face_missing_frames = max(1, int(self.face_missing_frames))
        self.multiple_face_frames = max(1, int(self.multiple_face_frames))
        self.face_outside_frames = max(1, int(self.face_outside_frames))
        self.look_away_frames = max(1, int(self.look_away_frames))
        self.look_away_ratio_threshold = _clamp(self.look_away_ratio_threshold, 0.05, 0.45)
        self.head_yaw_threshold = _clamp(self.head_yaw_threshold, 8.0, 60.0)
        self.head_pitch_threshold = _clamp(self.head_pitch_threshold, 8.0, 60.0)
        self.gaze_horizontal_threshold = _clamp(
            self.gaze_horizontal_threshold, 0.05, 0.50
        )
        self.gaze_vertical_threshold = _clamp(
            self.gaze_vertical_threshold, 0.05, 0.50
        )
        self.eye_closed_threshold = _clamp(self.eye_closed_threshold, 0.08, 0.40)
        self.edge_margin_ratio = _clamp(self.edge_margin_ratio, 0.0, 0.20)
        self.face_tracking_grace_frames = max(
            0, min(15, int(self.face_tracking_grace_frames))
        )

        self.hand_movement_threshold = max(1.0, float(self.hand_movement_threshold))
        self.hand_movement_ratio_threshold = _clamp(
            self.hand_movement_ratio_threshold, 0.02, 1.0
        )
        self.excessive_hand_frames = max(1, int(self.excessive_hand_frames))
        self.hand_cover_frames = max(1, int(self.hand_cover_frames))
        self.hand_cover_min_landmarks = max(
            1, min(21, int(self.hand_cover_min_landmarks))
        )
        self.face_movement_threshold = max(1.0, float(self.face_movement_threshold))
        self.face_movement_ratio_threshold = _clamp(
            self.face_movement_ratio_threshold, 0.02, 1.0
        )
        self.frequent_movement_frames = max(1, int(self.frequent_movement_frames))
        self.face_cover_overlap_ratio = _clamp(self.face_cover_overlap_ratio, 0.05, 1.0)
        self.suspicious_gesture_frames = max(1, int(self.suspicious_gesture_frames))
        self.gesture_history_frames = max(3, int(self.gesture_history_frames))
        self.gesture_stable_frames = max(
            1, min(self.gesture_history_frames, int(self.gesture_stable_frames))
        )
        self.gesture_majority_ratio = _clamp(
            self.gesture_majority_ratio, 0.50, 1.0
        )
        self.gesture_min_confidence = _clamp(
            self.gesture_min_confidence, 0.0, 1.0
        )
        self.minimum_face_area_ratio = _clamp(
            self.minimum_face_area_ratio, 0.001, 0.25
        )
        self.trained_gesture_min_confidence = _clamp(
            self.trained_gesture_min_confidence, 0.40, 1.0
        )

        duration_names = (
            "face_missing_seconds",
            "multiple_face_seconds",
            "face_outside_seconds",
            "look_away_seconds",
            "eyes_closed_seconds",
            "hand_cover_seconds",
            "excessive_hand_seconds",
            "frequent_movement_seconds",
            "suspicious_gesture_seconds",
        )
        for duration_name in duration_names:
            duration_value = _clamp(getattr(self, duration_name), 0.1, 30.0)
            setattr(self, duration_name, duration_value)

        allowed_levels = {"INFO", "WARNING", "CRITICAL"}
        self.evidence_minimum_level = str(self.evidence_minimum_level).upper()
        if self.evidence_minimum_level not in allowed_levels:
            self.evidence_minimum_level = "WARNING"
        self.risk_decay_per_second = _clamp(
            self.risk_decay_per_second, 0.0, 20.0
        )
        self.risk_warning_threshold = _clamp(
            self.risk_warning_threshold, 5.0, 90.0
        )
        self.risk_critical_threshold = _clamp(
            self.risk_critical_threshold,
            self.risk_warning_threshold + 1.0,
            100.0,
        )
        self.alert_cooldown_seconds = max(0.0, float(self.alert_cooldown_seconds))
        return self


def _clamp(value, minimum, maximum):
    """Return a number that stays between the given minimum and maximum."""
    return max(minimum, min(maximum, float(value)))


def load_settings(settings_file=SETTINGS_FILE):
    """Load settings from JSON and use defaults when the file is missing or invalid."""
    settings_path = Path(settings_file)
    settings = AppSettings()

    if not settings_path.exists():
        return settings.validate()

    try:
        saved_values = json.loads(settings_path.read_text(encoding="utf-8"))
        valid_names = {field.name for field in fields(AppSettings)}
        filtered_values = {
            name: value for name, value in saved_values.items() if name in valid_names
        }
        settings = AppSettings(**filtered_values)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        settings = AppSettings()

    return settings.validate()


def save_settings(settings, settings_file=SETTINGS_FILE):
    """Validate and save settings as readable JSON."""
    settings.validate()
    settings_path = Path(settings_file)
    settings_path.write_text(
        json.dumps(asdict(settings), indent=4, sort_keys=True),
        encoding="utf-8",
    )
    return settings_path


# CustomTkinter appearance values used across the dashboard.
WINDOW_TITLE = "AI Exam Monitoring System"
THEME = "dark"
COLOR_SCHEME = "blue"
FEED_WIDTH = 720
FEED_HEIGHT = 540

COLOR_OK = "#22c55e"
COLOR_WARNING = "#f59e0b"
COLOR_CRITICAL = "#ef4444"
COLOR_INFO = "#3b82f6"
COLOR_NEUTRAL = "#64748b"

GESTURE_LABELS = (
    "Open Palm",
    "Closed Fist",
    "Pointing Finger",
    "Victory Sign",
    "Thumbs Up",
    "Thumbs Down",
    "Phone Gesture",
    "Unknown Gesture",
)

SUSPICIOUS_GESTURES = {"Phone Gesture", "Victory Sign"}
