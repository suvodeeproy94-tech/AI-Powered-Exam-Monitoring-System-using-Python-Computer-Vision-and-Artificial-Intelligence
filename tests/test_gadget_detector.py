"""Tests for simple digital gadget detection."""

import unittest

import cv2
import numpy as np

from config import AppSettings
from detection.gadget_detector import GadgetDetector


class GadgetDetectorTests(unittest.TestCase):
    """Verify phone-like rectangles are detected without using a webcam."""

    def test_phone_like_rectangle_is_detected(self):
        settings = AppSettings(
            digital_gadget_min_confidence=0.45,
            digital_gadget_min_area_ratio=0.005,
        )
        detector = GadgetDetector(settings)
        frame = np.full((480, 640, 3), 230, dtype=np.uint8)

        cv2.rectangle(frame, (260, 120), (340, 300), (20, 20, 20), -1)
        cv2.rectangle(frame, (275, 145), (325, 265), (70, 70, 70), -1)

        _, results = detector.process_frame(frame)

        self.assertEqual(results["gadget_count"], 1)
        self.assertTrue(results["gadget_detected"])
        self.assertGreaterEqual(results["gadget_confidences"][0], 0.45)

    def test_plain_frame_has_no_gadget(self):
        detector = GadgetDetector(AppSettings())
        frame = np.full((480, 640, 3), 230, dtype=np.uint8)

        _, results = detector.process_frame(frame)

        self.assertEqual(results["gadget_count"], 0)
        self.assertFalse(results["gadget_detected"])


if __name__ == "__main__":
    unittest.main()
