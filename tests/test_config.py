"""Tests for saving, loading, and validating application settings."""

import tempfile
import unittest
from pathlib import Path

from config import AppSettings, load_settings, save_settings


class AppSettingsTests(unittest.TestCase):
    """Verify settings remain safe and can be stored as JSON."""

    def test_validation_keeps_values_inside_safe_limits(self):
        settings = AppSettings(
            camera_width=100,
            camera_height=5000,
            face_detection_confidence=4.0,
            yunet_score_threshold=0.1,
            alert_cooldown_seconds=-2,
        ).validate()

        self.assertEqual(settings.camera_width, 320)
        self.assertEqual(settings.camera_height, 1080)
        self.assertEqual(settings.face_detection_confidence, 1.0)
        self.assertEqual(settings.yunet_score_threshold, 0.30)
        self.assertEqual(settings.alert_cooldown_seconds, 0.0)

    def test_save_and_load_settings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings_path = Path(temporary_directory) / "settings.json"
            original_settings = AppSettings(
                camera_index=2,
                camera_width=1280,
                camera_height=720,
                logging_enabled=False,
            )

            save_settings(original_settings, settings_path)
            loaded_settings = load_settings(settings_path)

            self.assertEqual(loaded_settings.camera_index, 2)
            self.assertEqual(loaded_settings.camera_width, 1280)
            self.assertEqual(loaded_settings.camera_height, 720)
            self.assertFalse(loaded_settings.logging_enabled)


if __name__ == "__main__":
    unittest.main()
