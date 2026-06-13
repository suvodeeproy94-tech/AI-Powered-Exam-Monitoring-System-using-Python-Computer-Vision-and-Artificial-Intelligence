"""Integration test for local MediaPipe model loading and blank-frame handling."""

import unittest

import numpy as np

from config import AppSettings
from detection.face_detector import FaceDetector
from detection.hand_detector import HandDetector


class DetectionModelTests(unittest.TestCase):
    """Verify official model files load without requiring a physical webcam."""

    def test_models_process_a_blank_frame(self):
        settings = AppSettings()
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_detector = FaceDetector(settings)
        hand_detector = HandDetector(settings)

        try:
            face_frame, face_results = face_detector.process_frame(blank_frame)
            final_frame, hand_results = hand_detector.process_frame(
                blank_frame, face_frame
            )
        finally:
            face_detector.release()
            hand_detector.release()

        self.assertEqual(face_results["face_count"], 0)
        self.assertEqual(hand_results["hand_count"], 0)
        self.assertEqual(final_frame.shape, blank_frame.shape)


if __name__ == "__main__":
    unittest.main()
