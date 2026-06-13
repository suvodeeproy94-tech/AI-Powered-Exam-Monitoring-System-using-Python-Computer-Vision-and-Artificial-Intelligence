"""Tests for low-light camera preparation before AI detection."""

import unittest

import numpy as np

from config import AppSettings
from detection.frame_quality import prepare_detection_frame


class FrameQualityTests(unittest.TestCase):
    """Verify dark frames are enhanced and normal frames stay unchanged."""

    def setUp(self):
        self.settings = AppSettings(
            enhance_low_light=True,
            low_light_threshold=90.0,
            blur_threshold=45.0,
        )

    def test_dark_frame_is_marked_and_enhanced(self):
        dark_frame = np.full((120, 160, 3), 30, dtype=np.uint8)

        prepared_frame, quality = prepare_detection_frame(
            dark_frame, self.settings
        )

        self.assertTrue(quality["is_low_light"])
        self.assertTrue(quality["frame_was_enhanced"])
        self.assertEqual(prepared_frame.shape, dark_frame.shape)

    def test_normal_light_frame_is_not_changed(self):
        bright_frame = np.full((120, 160, 3), 180, dtype=np.uint8)

        prepared_frame, quality = prepare_detection_frame(
            bright_frame, self.settings
        )

        self.assertFalse(quality["is_low_light"])
        self.assertFalse(quality["frame_was_enhanced"])
        self.assertIs(prepared_frame, bright_frame)


if __name__ == "__main__":
    unittest.main()
