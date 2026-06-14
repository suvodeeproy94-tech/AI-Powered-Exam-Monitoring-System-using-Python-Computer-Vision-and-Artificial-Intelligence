"""Calculate one explainable suspicion score from confirmed exam events."""

import time


EVENT_WEIGHTS = {
    "FACE_MISSING": 30.0,
    "MULTIPLE_FACES": 50.0,
    "FACE_OUTSIDE_FRAME": 15.0,
    "LOOKING_AWAY": 12.0,
    "EYES_CLOSED": 8.0,
    "HAND_COVERING_FACE": 20.0,
    "EXCESSIVE_HAND_MOVEMENT": 10.0,
    "FREQUENT_MOVEMENT": 10.0,
    "SUSPICIOUS_GESTURE": 35.0,
}


class RiskScoreEngine:
    """Add event weights and slowly reduce the score during normal behavior."""

    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self, current_time=None):
        """Clear the current and maximum risk values."""
        self.score = 0.0
        self.maximum_score = 0.0
        self.last_update = time.monotonic() if current_time is None else current_time

    def update_time(self, current_time=None):
        """Apply time-based decay since the previous processed frame."""
        now = time.monotonic() if current_time is None else float(current_time)
        elapsed = max(0.0, now - self.last_update)
        self.score = max(
            0.0,
            self.score - elapsed * self.settings.risk_decay_per_second,
        )
        self.last_update = now
        return self.score

    def add_event(self, event_type):
        """Increase risk using the documented weight for one confirmed event."""
        self.score = min(100.0, self.score + EVENT_WEIGHTS.get(event_type, 5.0))
        self.maximum_score = max(self.maximum_score, self.score)
        return self.score

    def preview_event(self, event_type):
        """Return the score an event would create without changing state."""
        return min(100.0, self.score + EVENT_WEIGHTS.get(event_type, 5.0))

    def level(self):
        """Return NORMAL, WARNING, or CRITICAL from the current score."""
        if self.score >= self.settings.risk_critical_threshold:
            return "CRITICAL"
        if self.score >= self.settings.risk_warning_threshold:
            return "WARNING"
        return "NORMAL"

    def snapshot(self):
        """Return rounded score values for the dashboard and reports."""
        return {
            "risk_score": round(self.score, 1),
            "maximum_risk_score": round(self.maximum_score, 1),
            "risk_level": self.level(),
        }
