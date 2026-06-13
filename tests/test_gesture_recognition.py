"""Tests for the beginner-friendly rule-based gesture recognizer."""

import unittest

from recognition.gesture_recognition import GestureRecognizer


def build_landmarks(extended_fingers=None, thumb_mode="tucked"):
    """Create simple synthetic landmarks for one requested hand shape."""
    extended_fingers = set(extended_fingers or [])
    landmarks = [(100, 180) for _ in range(21)]
    landmarks[0] = (100, 220)

    finger_indexes = {
        "index": (6, 8, 80),
        "middle": (10, 12, 100),
        "ring": (14, 16, 120),
        "pinky": (18, 20, 140),
    }
    for finger_name, (pip_index, tip_index, x_position) in finger_indexes.items():
        landmarks[pip_index] = (x_position, 140)
        tip_y = 90 if finger_name in extended_fingers else 180
        landmarks[tip_index] = (x_position, tip_y)

    landmarks[2] = (75, 170)
    landmarks[3] = (65, 165)
    if thumb_mode == "outward":
        landmarks[4] = (35, 160)
    elif thumb_mode == "up":
        landmarks[2] = (90, 175)
        landmarks[3] = (90, 145)
        landmarks[4] = (90, 90)
    elif thumb_mode == "down":
        landmarks[2] = (90, 155)
        landmarks[3] = (90, 185)
        landmarks[4] = (90, 230)
    else:
        landmarks[4] = (85, 175)

    return landmarks


class GestureRecognizerTests(unittest.TestCase):
    """Verify every required named gesture can be identified."""

    def setUp(self):
        self.recognizer = GestureRecognizer()

    def assert_gesture(self, expected_name, landmarks):
        gesture_name, confidence, _ = self.recognizer.recognize(landmarks, "Right")
        self.assertEqual(gesture_name, expected_name)
        self.assertGreater(confidence, 0.5)

    def test_open_palm(self):
        self.assert_gesture(
            "Open Palm",
            build_landmarks({"index", "middle", "ring", "pinky"}, "outward"),
        )

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


if __name__ == "__main__":
    unittest.main()
