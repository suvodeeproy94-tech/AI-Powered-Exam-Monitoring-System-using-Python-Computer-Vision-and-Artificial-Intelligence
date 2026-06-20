"""Tests for the reference-style live camera overlay."""

import unittest

import numpy as np

from detection.hand_detector import HandDetector
from gui.video_overlay import (
    FACE_ALERT_COLOR,
    FACE_OK_COLOR,
    build_face_summary,
    build_gadget_summary,
    build_gesture_summary,
    draw_monitoring_overlay,
)


class VideoOverlayTests(unittest.TestCase):
    """Verify overlay wording, colors, and visible frame drawing."""

    def test_one_face_uses_reference_message_and_green_color(self):
        text, color = build_face_summary(1)
        self.assertEqual(text, "One face detected")
        self.assertEqual(color, FACE_OK_COLOR)

    def test_multiple_faces_use_alert_color(self):
        text, color = build_face_summary(2)
        self.assertEqual(text, "2 faces detected")
        self.assertEqual(color, FACE_ALERT_COLOR)

    def test_closed_fist_has_clear_gesture_text(self):
        gesture_results = [
            {
                "hand": "Right",
                "gesture": "Closed Fist",
                "confidence": 0.91,
            }
        ]
        text = build_gesture_summary(gesture_results)
        self.assertEqual(text, "Hand gesture: Closed Fist 91%")

    def test_unknown_gesture_shows_detecting_text(self):
        gesture_results = [
            {
                "hand": "Right",
                "gesture": "Unknown Gesture",
                "confidence": 0.0,
            }
        ]
        text = build_gesture_summary(gesture_results)
        self.assertEqual(text, "Hand gesture: detecting...")

    def test_gadget_summary_shows_possible_device(self):
        text = build_gadget_summary(
            {
                "gadget_count": 1,
                "gadget_confidences": [0.84],
            }
        )

        self.assertEqual(text, "Digital gadget: 1 possible 84%")

    def test_overlay_draws_visible_pixels(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        draw_monitoring_overlay(frame, 1, [])
        self.assertGreater(int(frame.sum()), 0)

    def test_hand_landmarks_use_red_points_and_white_bones(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        landmark_points = [(40 + index * 8, 120) for index in range(21)]
        detector = HandDetector.__new__(HandDetector)

        detector._draw_hand_landmarks(frame, landmark_points)

        point_blue, point_green, point_red = frame[120, 40]
        self.assertGreater(int(point_red), 200)
        self.assertLess(int(point_blue), 50)
        self.assertLess(int(point_green), 50)


if __name__ == "__main__":
    unittest.main()
