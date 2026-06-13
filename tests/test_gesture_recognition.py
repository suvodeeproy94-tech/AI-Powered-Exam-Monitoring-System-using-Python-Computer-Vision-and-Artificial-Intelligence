"""Tests for rotation-independent and stable hand gesture recognition."""

import math
import unittest

from config import AppSettings
from recognition.gesture_recognition import (
    GestureRecognizer,
    MultiHandGestureRecognizer,
)


FINGER_JOINTS = {
    "index": (5, 6, 7, 8, -1.5, 2.0),
    "middle": (9, 10, 11, 12, -0.5, 1.5),
    "ring": (13, 14, 15, 16, 0.5, 1.7),
    "pinky": (17, 18, 19, 20, 1.5, 2.2),
}


def build_landmarks(extended_fingers=None, thumb_mode="tucked"):
    """Create a complete synthetic hand with realistic joint bends."""
    extended_fingers = set(extended_fingers or [])
    landmarks = [(0.0, 0.0) for _ in range(21)]
    landmarks[0] = (0.0, 5.0)

    for finger_name, indexes in FINGER_JOINTS.items():
        mcp_index, pip_index, dip_index, tip_index, finger_x, mcp_y = indexes
        landmarks[mcp_index] = (finger_x, mcp_y)
        landmarks[pip_index] = (finger_x, mcp_y - 1.0)

        if finger_name in extended_fingers:
            landmarks[dip_index] = (finger_x, mcp_y - 2.0)
            landmarks[tip_index] = (finger_x, mcp_y - 3.0)
        else:
            landmarks[dip_index] = (finger_x + 0.8, mcp_y - 1.0)
            landmarks[tip_index] = (finger_x + 0.8, mcp_y)

    if thumb_mode == "outward":
        landmarks[1:5] = [
            (-0.8, 4.0),
            (-1.6, 3.5),
            (-2.4, 3.0),
            (-3.2, 2.5),
        ]
    elif thumb_mode == "up":
        landmarks[1:5] = [
            (-1.0, 4.0),
            (-1.0, 3.0),
            (-1.0, 2.0),
            (-1.0, 1.0),
        ]
    elif thumb_mode == "down":
        landmarks[1:5] = [
            (-1.0, 3.5),
            (-1.0, 4.5),
            (-1.0, 5.5),
            (-1.0, 6.5),
        ]
    else:
        landmarks[1:5] = [
            (-0.8, 4.0),
            (-1.4, 3.5),
            (-0.7, 3.5),
            (-0.3, 4.1),
        ]

    return landmarks


def rotate_landmarks(landmarks, degrees):
    """Rotate every point around the wrist to copy a sideways camera pose."""
    wrist_x, wrist_y = landmarks[0]
    angle = math.radians(degrees)
    rotated_points = []

    for point_x, point_y in landmarks:
        relative_x = point_x - wrist_x
        relative_y = point_y - wrist_y
        rotated_x = relative_x * math.cos(angle) - relative_y * math.sin(angle)
        rotated_y = relative_x * math.sin(angle) + relative_y * math.cos(angle)
        rotated_points.append((rotated_x + wrist_x, rotated_y + wrist_y))
    return rotated_points


class GestureRecognizerTests(unittest.TestCase):
    """Verify every required named gesture and the screenshot regression."""

    def setUp(self):
        self.recognizer = GestureRecognizer()

    def assert_gesture(self, expected_name, landmarks):
        """Check one expected gesture with a useful confidence value."""
        gesture_name, confidence, _ = self.recognizer.recognize(landmarks, "Right")
        self.assertEqual(gesture_name, expected_name)
        self.assertGreater(confidence, 0.5)

    def test_open_palm(self):
        self.assert_gesture(
            "Open Palm",
            build_landmarks({"index", "middle", "ring", "pinky"}, "outward"),
        )

    def test_rotated_open_palm_is_not_thumbs_up(self):
        open_palm = build_landmarks(
            {"index", "middle", "ring", "pinky"}, "outward"
        )
        self.assert_gesture("Open Palm", rotate_landmarks(open_palm, 78))

    def test_closed_fist(self):
        self.assert_gesture("Closed Fist", build_landmarks())

    def test_pointing_finger(self):
        self.assert_gesture("Pointing Finger", build_landmarks({"index"}))

    def test_victory_sign_is_suspicious(self):
        gesture_name, _, is_suspicious = self.recognizer.recognize(
            build_landmarks({"index", "middle"}), "Right"
        )
        self.assertEqual(gesture_name, "Victory Sign")
        self.assertTrue(is_suspicious)

    def test_thumbs_up(self):
        self.assert_gesture("Thumbs Up", build_landmarks(thumb_mode="up"))

    def test_thumbs_down(self):
        self.assert_gesture("Thumbs Down", build_landmarks(thumb_mode="down"))

    def test_phone_gesture_is_suspicious(self):
        gesture_name, _, is_suspicious = self.recognizer.recognize(
            build_landmarks({"pinky"}, "outward"), "Right"
        )
        self.assertEqual(gesture_name, "Phone Gesture")
        self.assertTrue(is_suspicious)

    def test_gesture_must_be_stable_before_display(self):
        settings = AppSettings(
            gesture_history_frames=5,
            gesture_stable_frames=3,
            gesture_majority_ratio=0.60,
        )
        recognizer = MultiHandGestureRecognizer(settings)
        open_palm = build_landmarks(
            {"index", "middle", "ring", "pinky"}, "outward"
        )

        first_result = recognizer.recognize_all([open_palm], ["Right"])[0]
        second_result = recognizer.recognize_all([open_palm], ["Right"])[0]
        third_result = recognizer.recognize_all([open_palm], ["Right"])[0]

        self.assertEqual(first_result["gesture"], "Unknown Gesture")
        self.assertEqual(second_result["gesture"], "Unknown Gesture")
        self.assertEqual(third_result["gesture"], "Open Palm")


if __name__ == "__main__":
    unittest.main()
