"""Save privacy-aware evidence images only for confirmed monitoring alerts."""

from datetime import datetime
from pathlib import Path
import re

import cv2

from config import EVIDENCE_DIR


LEVEL_ORDER = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}


class EvidenceManager:
    """Store one JPEG frame for alerts that meet the configured severity."""

    def __init__(self, settings, evidence_directory=EVIDENCE_DIR):
        self.settings = settings
        self.evidence_directory = Path(evidence_directory)

    def capture(self, frame, event_type, alert_level, timestamp=None):
        """Save a frame and return its path, or an empty string when disabled."""
        if not self.settings.evidence_capture_enabled or frame is None:
            return ""
        minimum_level = self.settings.evidence_minimum_level
        if LEVEL_ORDER.get(alert_level, 0) < LEVEL_ORDER.get(minimum_level, 1):
            return ""

        event_time = timestamp or datetime.now()
        day_directory = self.evidence_directory / event_time.strftime("%Y-%m-%d")
        day_directory.mkdir(parents=True, exist_ok=True)
        safe_event_name = re.sub(r"[^A-Z0-9_-]", "_", event_type.upper())
        file_name = f"{event_time:%H%M%S_%f}_{safe_event_name}.jpg"
        evidence_path = day_directory / file_name

        try:
            if not cv2.imwrite(str(evidence_path), frame):
                return ""
        except (cv2.error, OSError):
            return ""
        return str(evidence_path.resolve())
