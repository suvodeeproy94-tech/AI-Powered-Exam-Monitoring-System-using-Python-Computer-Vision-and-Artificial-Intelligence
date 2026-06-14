"""Build a personal neutral face and gaze profile before monitoring starts."""

from dataclasses import dataclass
import statistics
import time


@dataclass
class CalibrationResult:
    """Store calibration progress and the final neutral profile."""

    complete: bool
    progress: float
    message: str
    profile: dict


class CalibrationManager:
    """Collect good camera frames and calculate a student-specific baseline."""

    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self, current_time=None):
        """Start a new calibration session."""
        self.start_time = current_time
        self.total_frames = 0
        self.valid_frames = 0
        self.measurements = []
        self.complete = not self.settings.calibration_enabled
        self.profile = {}
        self.message = "Calibration disabled" if self.complete else "Look at the screen"

    def update(self, face_results, current_time=None):
        """Add one frame and return current calibration status."""
        now = time.monotonic() if current_time is None else float(current_time)
        if self.start_time is None:
            self.start_time = now
        if self.complete:
            return CalibrationResult(True, 1.0, self.message, dict(self.profile))

        self.total_frames += 1
        is_valid, message = self._frame_is_valid(face_results)
        self.message = message
        if is_valid:
            self.valid_frames += 1
            self.measurements.append(
                {
                    "head_yaw": face_results.get("head_yaw", 0.0),
                    "head_pitch": face_results.get("head_pitch", 0.0),
                    "gaze_horizontal": face_results.get("gaze_horizontal", 0.0),
                    "gaze_vertical": face_results.get("gaze_vertical", 0.0),
                    "face_area_ratio": face_results.get("primary_face_area_ratio", 0.0),
                }
            )

        elapsed = max(0.0, now - self.start_time)
        time_progress = min(1.0, elapsed / self.settings.calibration_seconds)
        valid_ratio = self.valid_frames / max(self.total_frames, 1)
        enough_valid_frames = self.valid_frames >= 10

        if (
            elapsed >= self.settings.calibration_seconds
            and valid_ratio >= self.settings.calibration_min_valid_ratio
            and enough_valid_frames
        ):
            self.profile = self._build_profile()
            self.complete = True
            self.message = "Calibration complete"
            return CalibrationResult(True, 1.0, self.message, dict(self.profile))

        # Keep the progress below 100% until enough good frames are available.
        if elapsed >= self.settings.calibration_seconds:
            time_progress = 0.99
        return CalibrationResult(
            False,
            time_progress,
            self.message,
            {},
        )

    def _frame_is_valid(self, face_results):
        """Check camera quality, face count, position, and face size."""
        if face_results.get("face_count", 0) != 1:
            return False, "Keep exactly one face in the camera"
        if face_results.get("is_low_light"):
            return False, "Increase the room lighting"
        if face_results.get("is_blurry"):
            return False, "Keep the camera and your face steady"
        if face_results.get("face_outside_frame"):
            return False, "Move your face to the center"

        face_area_ratio = face_results.get("primary_face_area_ratio", 0.0)
        if face_area_ratio < self.settings.minimum_face_size_ratio:
            return False, "Move closer to the camera"
        if face_area_ratio > self.settings.maximum_face_size_ratio:
            return False, "Move slightly away from the camera"
        if face_results.get("eyes_closed"):
            return False, "Keep your eyes open and look at the screen"
        return True, "Look at the screen and remain still"

    def _build_profile(self):
        """Use medians so one unusual frame does not distort calibration."""
        profile = {}
        for measurement_name in self.measurements[0]:
            values = [item[measurement_name] for item in self.measurements]
            profile[measurement_name] = round(statistics.median(values), 4)
        return profile
