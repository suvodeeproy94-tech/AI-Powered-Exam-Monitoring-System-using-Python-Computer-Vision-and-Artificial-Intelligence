"""Tests for stable suspicious-activity confirmation rules."""

import unittest

from config import AppSettings
from monitoring.alert_manager import AlertLevel, AlertManager
from monitoring.behavior_monitor import BehaviorMonitor, boxes_overlap_ratio


def normal_face_results():
    """Return one normal face result for behavior tests."""
    return {
        "face_count": 1,
        "face_bboxes": [(100, 100, 120, 120)],
        "face_outside_frame": False,
        "is_looking_away": False,
        "face_movement": 0.0,
    }


def normal_hand_results():
    """Return an empty hand result for behavior tests."""
    return {
        "hand_count": 0,
        "hand_bboxes": [],
        "hand_labels": [],
        "excessive_movement": False,
    }


class BehaviorMonitorTests(unittest.TestCase):
    """Verify alerts wait for the configured number of stable frames."""

    def setUp(self):
        self.settings = AppSettings(
            face_missing_frames=2,
            multiple_face_frames=2,
            face_outside_frames=2,
            look_away_frames=2,
            hand_cover_frames=2,
            excessive_hand_frames=2,
            frequent_movement_frames=2,
            suspicious_gesture_frames=2,
            alert_cooldown_seconds=0,
        )
        self.alert_manager = AlertManager(
            cooldown_seconds=0,
            logging_enabled=False,
        )
        self.monitor = BehaviorMonitor(self.alert_manager, self.settings)

    def test_missing_face_waits_for_confirmation(self):
        missing_face = normal_face_results()
        missing_face["face_count"] = 0
        missing_face["face_bboxes"] = []

        first_frame_alerts = self.monitor.analyse(
            missing_face, normal_hand_results(), []
        )
        second_frame_alerts = self.monitor.analyse(
            missing_face, normal_hand_results(), []
        )

        self.assertEqual(first_frame_alerts, [])
        self.assertEqual(len(second_frame_alerts), 1)
        self.assertEqual(second_frame_alerts[0].level, AlertLevel.CRITICAL)
        self.assertEqual(second_frame_alerts[0].event_type, "FACE_MISSING")

    def test_multiple_faces_are_critical(self):
        multiple_faces = normal_face_results()
        multiple_faces["face_count"] = 2
        multiple_faces["face_bboxes"] = [
            (50, 50, 100, 100),
            (250, 50, 100, 100),
        ]

        self.monitor.analyse(multiple_faces, normal_hand_results(), [])
        alerts = self.monitor.analyse(multiple_faces, normal_hand_results(), [])

        self.assertEqual(alerts[-1].event_type, "MULTIPLE_FACES")
        self.assertEqual(self.monitor.alert_status, "CRITICAL")

    def test_hand_covering_face_is_detected(self):
        hand_results = normal_hand_results()
        hand_results["hand_count"] = 1
        hand_results["hand_bboxes"] = [(110, 110, 90, 90)]

        self.monitor.analyse(normal_face_results(), hand_results, [])
        alerts = self.monitor.analyse(normal_face_results(), hand_results, [])

        event_types = [alert.event_type for alert in alerts]
        self.assertIn("HAND_COVERING_FACE", event_types)

    def test_suspicious_gesture_needs_two_frames(self):
        gesture_results = [
            {
                "hand": "Right",
                "gesture": "Phone Gesture",
                "confidence": 0.86,
                "is_suspicious": True,
            }
        ]

        first_alerts = self.monitor.analyse(
            normal_face_results(), normal_hand_results(), gesture_results
        )
        second_alerts = self.monitor.analyse(
            normal_face_results(), normal_hand_results(), gesture_results
        )

        self.assertNotIn(
            "SUSPICIOUS_GESTURE", [alert.event_type for alert in first_alerts]
        )
        self.assertIn(
            "SUSPICIOUS_GESTURE", [alert.event_type for alert in second_alerts]
        )

    def test_box_overlap_uses_face_area(self):
        overlap = boxes_overlap_ratio((0, 0, 100, 100), (0, 0, 50, 100))
        self.assertEqual(overlap, 0.5)


if __name__ == "__main__":
    unittest.main()
