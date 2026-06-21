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
            digital_gadget_min_area_ratio=0.005,
        )
        detector = GadgetDetector(settings)
        frame = np.full((480, 640, 3), 230, dtype=np.uint8)

        cv2.rectangle(frame, (260, 120), (340, 300), (20, 20, 20), -1)
        cv2.rectangle(frame, (275, 145), (325, 265), (70, 70, 70), -1)

        _, results = detector.process_frame(frame)

        self.assertEqual(results["gadget_count"], 1)
        self.assertTrue(results["gadget_detected"])
        self.assertGreaterEqual(results["gadget_confidences"][0], 0.72)

    def test_bright_blue_phone_held_in_hand_is_detected(self):
        settings = AppSettings(
            digital_gadget_min_area_ratio=0.005,
        )
        detector = GadgetDetector(settings)
        frame = np.full((480, 640, 3), 215, dtype=np.uint8)

        # This simulates a teal/blue phone like the one in the live screenshot.
        cv2.rectangle(frame, (180, 95), (285, 360), (180, 145, 0), -1)
        cv2.rectangle(frame, (190, 105), (275, 350), (210, 170, 0), -1)
        # A hand holding the lower part should not make the phone disappear.
        cv2.rectangle(frame, (145, 310), (240, 430), (75, 110, 155), -1)

        _, results = detector.process_frame(frame)

        self.assertGreaterEqual(results["gadget_count"], 1)
        self.assertTrue(results["gadget_detected"])

    def test_skin_colored_rectangle_is_not_detected_as_gadget(self):
        settings = AppSettings(
            digital_gadget_min_area_ratio=0.005,
        )
        detector = GadgetDetector(settings)
        frame = np.full((480, 640, 3), 220, dtype=np.uint8)

        # This represents a skin-colored arm or shoulder shape, not a phone.
        cv2.rectangle(frame, (250, 170), (360, 330), (70, 105, 150), -1)
        cv2.rectangle(frame, (265, 190), (345, 310), (80, 115, 165), -1)

        _, results = detector.process_frame(frame)

        self.assertEqual(results["gadget_count"], 0)
        self.assertFalse(results["gadget_detected"])

    def test_plain_frame_has_no_gadget(self):
        detector = GadgetDetector(AppSettings())
        frame = np.full((480, 640, 3), 230, dtype=np.uint8)

        _, results = detector.process_frame(frame)

        self.assertEqual(results["gadget_count"], 0)
        self.assertFalse(results["gadget_detected"])


if __name__ == "__main__":
    unittest.main()
