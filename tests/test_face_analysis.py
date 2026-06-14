"""Tests for detailed face analysis and stable primary-face tracking."""

import unittest

from config import AppSettings
from detection.face_analyzer import FaceAnalyzer
from detection.face_detector import FaceDetector


class FaceAnalysisTests(unittest.TestCase):
    """Verify safe empty results, pitch normalization, and tracking IDs."""

    def test_missing_landmarks_return_complete_defaults(self):
        analyzer = FaceAnalyzer(AppSettings())
        results = analyzer.analyse(None, 640, 480)

        self.assertFalse(results["head_pose_available"])
        self.assertFalse(results["gaze_available"])
        self.assertEqual(results["gaze_direction"], "Unknown")

    def test_rear_form_pitch_is_normalized(self):
        analyzer = FaceAnalyzer(AppSettings())
        self.assertAlmostEqual(analyzer._normalize_pitch(-160.0), -20.0)
        self.assertAlmostEqual(analyzer._normalize_pitch(160.0), 20.0)

    def test_primary_face_keeps_id_during_short_miss(self):
        detector = FaceDetector.__new__(FaceDetector)
        detector.settings = AppSettings(face_tracking_grace_frames=2)
        detector._missed_face_frames = 0
        detector._last_primary_face_box = None
        detector._next_track_id = 1
        detector._active_track_id = None

        visible, _, first_id = detector._update_face_tracking((10, 10, 100, 100))
        missed_visible, _, missed_id = detector._update_face_tracking(None)
        visible_again, _, second_id = detector._update_face_tracking((12, 10, 100, 100))

        self.assertTrue(visible)
        self.assertTrue(missed_visible)
        self.assertTrue(visible_again)
        self.assertEqual(first_id, missed_id)
        self.assertEqual(first_id, second_id)


if __name__ == "__main__":
    unittest.main()
