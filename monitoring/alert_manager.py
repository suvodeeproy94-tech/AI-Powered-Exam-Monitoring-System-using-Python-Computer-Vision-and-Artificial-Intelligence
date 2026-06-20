"""Create, store, display, and log monitoring alerts.

AlertManager is the only module that writes alert records. This keeps the CSV
format consistent for the dashboard and daily report generator.
"""

import csv
from dataclasses import dataclass, field
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading
import time

from config import (
    ACTIVITY_LOG_FILE,
    COLOR_CRITICAL,
    COLOR_INFO,
    COLOR_WARNING,
    SYSTEM_LOG_FILE,
)


class AlertLevel:
    """Allowed alert severity labels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


LEVEL_COLORS = {
    AlertLevel.INFO: COLOR_INFO,
    AlertLevel.WARNING: COLOR_WARNING,
    AlertLevel.CRITICAL: COLOR_CRITICAL,
}

CSV_FIELDS = [
    "session_id",
    "student_name",
    "roll_number",
    "exam_name",
    "subject_name",
    "date",
    "time",
    "event_type",
    "alert_type",
    "description",
    "duration_seconds",
    "risk_score",
    "attention_percentage",
    "evidence_path",
]


def _create_logger():
    """Create one console and rotating-file logger for application messages."""
    application_logger = logging.getLogger("ExamMonitor")
    if application_logger.handlers:
        return application_logger

    application_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    application_logger.addHandler(console_handler)

    try:
        file_handler = RotatingFileHandler(
            SYSTEM_LOG_FILE,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        application_logger.addHandler(file_handler)
    except OSError:
        application_logger.warning("System log file could not be created.")

    return application_logger


logger = _create_logger()


@dataclass(frozen=True)
class Alert:
    """Represent one alert shown in the GUI and written to the CSV log."""

    level: str
    event_type: str
    description: str
    session_id: str = ""
    student_name: str = ""
    roll_number: str = ""
    exam_name: str = ""
    subject_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    risk_score: float = 0.0
    attention_percentage: float = 100.0
    evidence_path: str = ""

    def to_dict(self):
        """Convert the alert into the exact activity CSV column format."""
        return {
            "session_id": self.session_id,
            "student_name": self.student_name,
            "roll_number": self.roll_number,
            "exam_name": self.exam_name,
            "subject_name": self.subject_name,
            "date": self.timestamp.strftime("%Y-%m-%d"),
            "time": self.timestamp.strftime("%H:%M:%S"),
            "event_type": self.event_type,
            "alert_type": self.level,
            "description": self.description,
            "duration_seconds": f"{self.duration_seconds:.2f}",
            "risk_score": f"{self.risk_score:.1f}",
            "attention_percentage": f"{self.attention_percentage:.1f}",
            "evidence_path": self.evidence_path,
        }

    def __str__(self):
        """Return a short human-readable alert line."""
        return (
            f"[{self.timestamp:%H:%M:%S}] {self.level:8s} | "
            f"{self.event_type}: {self.description}"
        )


class AlertManager:
    """Apply cooldown rules and keep one thread-safe alert history."""

    def __init__(
        self,
        log_file=ACTIVITY_LOG_FILE,
        cooldown_seconds=5.0,
        logging_enabled=True,
        max_history=200,
    ):
        self.log_file = Path(log_file)
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.logging_enabled = bool(logging_enabled)
        self.max_history = max(1, int(max_history))
        self.alert_history = []
        self._last_alert_time = {}
        self._current_session = None
        self._session_manager = None
        self._lock = threading.RLock()

        if self.logging_enabled:
            self._ensure_log_file()

    def _ensure_log_file(self):
        """Create the activity CSV and header when they do not exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.log_file.exists() and self.log_file.stat().st_size > 0:
            self._upgrade_old_log_columns()
            return

        with self.log_file.open("w", newline="", encoding="utf-8") as csv_file:
            csv.DictWriter(csv_file, fieldnames=CSV_FIELDS).writeheader()

    def _upgrade_old_log_columns(self):
        """Add new optional columns to logs created by an older project version."""
        try:
            with self.log_file.open("r", newline="", encoding="utf-8-sig") as csv_file:
                reader = csv.DictReader(csv_file)
                if reader.fieldnames == CSV_FIELDS:
                    return
                old_rows = list(reader)

            with self.log_file.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for old_row in old_rows:
                    writer.writerow(
                        {
                            field_name: old_row.get(field_name, "")
                            for field_name in CSV_FIELDS
                        }
                    )
        except OSError as error:
            logger.error("Old activity log could not be upgraded: %s", error)

    def _write_to_csv(self, alert):
        """Append one alert row without exposing logging errors to the GUI."""
        if not self.logging_enabled:
            return

        try:
            with self.log_file.open("a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                writer.writerow(alert.to_dict())
        except OSError as error:
            logger.error("Activity log could not be updated: %s", error)

    def set_session_manager(self, session_manager):
        """Attach the optional SQLite session storage."""
        with self._lock:
            self._session_manager = session_manager

    def set_current_session(self, session):
        """Attach student and exam details to future alerts."""
        with self._lock:
            self._current_session = session

    def clear_current_session(self):
        """Remove session details after the monitoring session ends."""
        with self._lock:
            self._current_session = None

    def _session_metadata(self):
        """Return safe session values for one alert record."""
        session = self._current_session
        if session is None:
            return {
                "session_id": "",
                "student_name": "",
                "roll_number": "",
                "exam_name": "",
                "subject_name": "",
            }
        return {
            "session_id": session.session_id,
            "student_name": session.student_name,
            "roll_number": session.roll_number,
            "exam_name": session.exam_name,
            "subject_name": session.subject_name,
        }

    def _is_in_cooldown(self, event_type, current_time):
        """Check whether the same event was recently recorded."""
        last_alert_time = self._last_alert_time.get(event_type, 0.0)
        return current_time - last_alert_time < self.cooldown_seconds

    def can_fire(self, event_type, force=False):
        """Check cooldown before doing optional work such as evidence capture."""
        if force:
            return True
        with self._lock:
            return not self._is_in_cooldown(event_type, time.monotonic())

    def fire(
        self,
        level,
        event_type,
        description,
        force=False,
        duration_seconds=0.0,
        risk_score=0.0,
        attention_percentage=100.0,
        evidence_path="",
    ):
        """Create an alert unless its event type is still in cooldown."""
        if level not in (AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL):
            raise ValueError(f"Unknown alert level: {level}")
        if not event_type or not description:
            raise ValueError("Alert event type and description are required.")

        current_time = time.monotonic()
        with self._lock:
            if not force and self._is_in_cooldown(event_type, current_time):
                return None

            alert = Alert(
                level=level,
                event_type=event_type,
                description=description,
                **self._session_metadata(),
                duration_seconds=max(0.0, float(duration_seconds)),
                risk_score=max(0.0, min(100.0, float(risk_score))),
                attention_percentage=max(
                    0.0, min(100.0, float(attention_percentage))
                ),
                evidence_path=str(evidence_path or ""),
            )
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history:
                self.alert_history = self.alert_history[-self.max_history :]

            self._last_alert_time[event_type] = current_time
            self._write_to_csv(alert)
            if self._session_manager is not None:
                self._session_manager.store_alert(alert)

        log_function = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }[level]
        log_function("%s: %s", event_type, description)
        return alert

    def info(self, event_type, description, force=False, **metadata):
        """Create an information alert."""
        return self.fire(
            AlertLevel.INFO, event_type, description, force, **metadata
        )

    def warning(self, event_type, description, force=False, **metadata):
        """Create a warning alert."""
        return self.fire(
            AlertLevel.WARNING, event_type, description, force, **metadata
        )

    def critical(self, event_type, description, force=False, **metadata):
        """Create a critical alert."""
        return self.fire(
            AlertLevel.CRITICAL, event_type, description, force, **metadata
        )

    def get_history(self):
        """Return a safe copy for the GUI thread."""
        with self._lock:
            return list(self.alert_history)

    def summary(self):
        """Count in-memory alerts by severity."""
        counts = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 0,
            AlertLevel.CRITICAL: 0,
        }
        with self._lock:
            for alert in self.alert_history:
                counts[alert.level] += 1
        return counts

    def clear_history(self):
        """Clear only the dashboard history and preserve the CSV audit log."""
        with self._lock:
            self.alert_history.clear()
            self._last_alert_time.clear()

    def set_logging_enabled(self, is_enabled):
        """Turn CSV activity logging on or off at runtime."""
        with self._lock:
            self.logging_enabled = bool(is_enabled)
            if self.logging_enabled:
                self._ensure_log_file()
