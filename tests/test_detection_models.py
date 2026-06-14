"""Integration test for local MediaPipe model loading and blank-frame handling."""

import unittest

import numpy as np

from config import AppSettings, YUNET_FACE_DETECTOR_MODEL
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
        self.assertEqual(face_results["face_detector_backend"], "YuNet")
        self.assertEqual(hand_results["hand_count"], 0)
        self.assertEqual(final_frame.shape, blank_frame.shape)

    def test_yunet_model_is_bundled_and_loads(self):
        """Verify the downloaded ONNX model is part of the project."""
        self.assertTrue(YUNET_FACE_DETECTOR_MODEL.exists())
        detector = FaceDetector(AppSettings())
        try:
            self.assertIsNotNone(detector.yunet_detector)
        finally:
            detector.release()

    def test_mediapipe_fallback_can_be_selected(self):
        """Keep the original face detector available as a safe fallback."""
        settings = AppSettings(yunet_face_detection_enabled=False)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = FaceDetector(settings)
        try:
            _, results = detector.process_frame(blank_frame)
        finally:
            detector.release()

        self.assertEqual(results["face_detector_backend"], "MediaPipe")


if __name__ == "__main__":
    unittest.main()
