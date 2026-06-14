"""Tests for personal camera, head pose, and gaze calibration."""

import unittest

from config import AppSettings
from monitoring.calibration_manager import CalibrationManager


def valid_face_results():
    """Return one suitable calibration frame."""
    return {
        "face_count": 1,
        "is_low_light": False,
        "is_blurry": False,
        "face_outside_frame": False,
        "primary_face_area_ratio": 0.15,
        "eyes_closed": False,
        "head_yaw": 4.0,
        "head_pitch": -2.0,
        "gaze_horizontal": 0.05,
        "gaze_vertical": -0.03,
    }


class CalibrationManagerTests(unittest.TestCase):
    """Verify quality instructions and neutral profile creation."""

    def test_calibration_builds_median_profile(self):
        settings = AppSettings(
            calibration_seconds=1.0,
            calibration_min_valid_ratio=0.5,
        )
        manager = CalibrationManager(settings)
        manager.reset(current_time=0.0)

        result = None
        for frame_index in range(12):
            result = manager.update(
                valid_face_results(), current_time=frame_index * 0.1
            )

        self.assertTrue(result.complete)
        self.assertEqual(result.profile["head_yaw"], 4.0)
        self.assertEqual(result.profile["gaze_horizontal"], 0.05)

    def test_small_face_requests_moving_closer(self):
        manager = CalibrationManager(AppSettings())
        face_results = valid_face_results()
        face_results["primary_face_area_ratio"] = 0.01

        result = manager.update(face_results, current_time=0.0)

        self.assertFalse(result.complete)
        self.assertEqual(result.message, "Move closer to the camera")

    def test_incomplete_calibration_does_not_show_full_progress(self):
        """Show 99% until enough valid camera frames have been collected."""
        settings = AppSettings(calibration_seconds=1.0)
        manager = CalibrationManager(settings)
        manager.reset(current_time=0.0)
        invalid_face_results = valid_face_results()
        invalid_face_results["face_count"] = 0

        result = manager.update(invalid_face_results, current_time=1.2)

        self.assertFalse(result.complete)
        self.assertEqual(result.progress, 0.99)


if __name__ == "__main__":
    unittest.main()
