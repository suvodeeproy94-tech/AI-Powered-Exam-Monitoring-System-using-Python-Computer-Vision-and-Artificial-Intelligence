"""Tests for trainable gesture feature normalization."""

import unittest

from recognition.gesture_features import extract_gesture_features
from tests.test_gesture_recognition import build_landmarks, rotate_landmarks


class GestureFeatureTests(unittest.TestCase):
    """Verify feature size and rotation stability for model training."""

    def test_features_have_expected_size(self):
        landmarks = build_landmarks({"index", "middle", "ring", "pinky"})
        self.assertEqual(len(extract_gesture_features(landmarks)), 42)

    def test_rotated_pose_has_similar_features(self):
        landmarks = build_landmarks({"index", "middle", "ring", "pinky"})
        rotated = rotate_landmarks(landmarks, 60)
        first_features = extract_gesture_features(landmarks)
        rotated_features = extract_gesture_features(rotated)

        maximum_difference = max(
            abs(first - second)
            for first, second in zip(first_features, rotated_features)
        )
        self.assertLess(maximum_difference, 0.001)


if __name__ == "__main__":
    unittest.main()
