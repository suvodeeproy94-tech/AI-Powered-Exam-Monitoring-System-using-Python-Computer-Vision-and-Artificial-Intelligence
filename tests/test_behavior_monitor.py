"""Tests for stable suspicious-activity confirmation rules."""

import unittest

from config import AppSettings
from monitoring.alert_manager import AlertLevel, AlertManager
from monitoring.behavior_monitor import (
    BehaviorMonitor,
    boxes_overlap_ratio,
    count_landmarks_inside_box,
)


def normal_face_results():
    """Return one normal face result for behavior tests."""
    return {
        "face_count": 1,
        "face_bboxes": [(100, 100, 120, 120)],
        "face_outside_frame": False,
        "is_looking_away": False,
        "face_movement": 0.0,
        "face_movement_ratio": 0.0,
    }


def normal_hand_results():
    """Return an empty hand result for behavior tests."""
    return {
        "hand_count": 0,
        "hand_bboxes": [],
        "hand_labels": [],
        "hand_landmark_points": [],
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
            digital_gadget_frames=2,
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
        hand_results["hand_landmark_points"] = [
            [(130, 130) for _ in range(21)]
        ]

        self.monitor.analyse(normal_face_results(), hand_results, [])
        alerts = self.monitor.analyse(normal_face_results(), hand_results, [])

        event_types = [alert.event_type for alert in alerts]
        self.assertIn("HAND_COVERING_FACE", event_types)

    def test_overlapping_box_without_face_landmarks_is_not_a_cover(self):
        hand_results = normal_hand_results()
        hand_results["hand_count"] = 1
        hand_results["hand_bboxes"] = [(110, 110, 90, 90)]
        hand_results["hand_landmark_points"] = [
            [(260, 260) for _ in range(21)]
        ]

        self.monitor.analyse(normal_face_results(), hand_results, [])
        alerts = self.monitor.analyse(normal_face_results(), hand_results, [])

        event_types = [alert.event_type for alert in alerts]
        self.assertNotIn("HAND_COVERING_FACE", event_types)

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

    def test_digital_gadget_needs_two_frames(self):
        gadget_results = {
            "gadget_count": 1,
            "gadget_boxes": [(260, 120, 80, 180)],
            "gadget_confidences": [0.82],
            "gadget_detected": True,
        }

        first_alerts = self.monitor.analyse(
            normal_face_results(),
            normal_hand_results(),
            [],
            gadget_results,
        )
        second_alerts = self.monitor.analyse(
            normal_face_results(),
            normal_hand_results(),
            [],
            gadget_results,
        )

        self.assertNotIn(
            "DIGITAL_GADGET_DETECTED",
            [alert.event_type for alert in first_alerts],
        )
        self.assertIn(
            "DIGITAL_GADGET_DETECTED",
            [alert.event_type for alert in second_alerts],
        )
        self.assertEqual(self.monitor.stats["digital_gadget_violations"], 1)

    def test_box_overlap_uses_face_area(self):
        overlap = boxes_overlap_ratio((0, 0, 100, 100), (0, 0, 50, 100))
        self.assertEqual(overlap, 0.5)

    def test_landmarks_inside_face_box_are_counted(self):
        landmark_points = [(20, 20), (40, 40), (120, 120)]
        count = count_landmarks_inside_box(landmark_points, (0, 0, 100, 100))
        self.assertEqual(count, 2)

    def test_runtime_confirmation_uses_seconds_not_frame_count(self):
        self.settings.face_missing_frames = 1
        self.settings.face_missing_seconds = 2.0
        missing_face = normal_face_results()
        missing_face["face_count"] = 0
        missing_face["face_bboxes"] = []

        first_alerts = self.monitor.analyse(
            missing_face, normal_hand_results(), [], current_time=0.0
        )
        early_alerts = self.monitor.analyse(
            missing_face, normal_hand_results(), [], current_time=1.5
        )
        confirmed_alerts = self.monitor.analyse(
            missing_face, normal_hand_results(), [], current_time=2.1
        )

        self.assertEqual(first_alerts, [])
        self.assertEqual(early_alerts, [])
        self.assertIn(
            "FACE_MISSING", [alert.event_type for alert in confirmed_alerts]
        )

    def test_normal_blink_does_not_create_eye_closed_alert(self):
        closed_eyes = normal_face_results()
        closed_eyes["eyes_closed"] = True
        self.settings.eyes_closed_seconds = 3.0

        first_alerts = self.monitor.analyse(
            closed_eyes, normal_hand_results(), [], current_time=0.0
        )
        blink_alerts = self.monitor.analyse(
            closed_eyes, normal_hand_results(), [], current_time=0.4
        )

        self.assertNotIn(
            "EYES_CLOSED",
            [alert.event_type for alert in first_alerts + blink_alerts],
        )

    def test_repeated_alert_logs_only_new_condition_duration(self):
        """Avoid counting the same continuous incident time more than once."""
        missing_face = normal_face_results()
        missing_face["face_count"] = 0
        missing_face["face_bboxes"] = []
        self.settings.face_missing_seconds = 1.0

        self.monitor.analyse(
            missing_face, normal_hand_results(), [], current_time=0.0
        )
        first_alerts = self.monitor.analyse(
            missing_face, normal_hand_results(), [], current_time=1.2
        )
        repeated_alerts = self.monitor.analyse(
            missing_face, normal_hand_results(), [], current_time=2.0
        )

        self.assertAlmostEqual(first_alerts[0].duration_seconds, 1.2)
        self.assertAlmostEqual(repeated_alerts[0].duration_seconds, 0.8)


if __name__ == "__main__":
    unittest.main()
