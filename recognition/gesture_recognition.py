"""Recognise common exam-monitoring hand gestures from 21 landmarks.

This module uses clear geometric rules instead of a trained model. The rules
are fast enough for live webcam use and easy to explain during a viva.
"""

import math

from config import GESTURE_LABELS, SUSPICIOUS_GESTURES


def _distance(first_point, second_point):
    """Calculate the straight-line distance between two pixel points."""
    return math.dist(first_point, second_point)


def _finger_is_extended(landmarks, tip_index, pip_index):
    """Check whether a finger tip is above its middle joint on the image."""
    return landmarks[tip_index][1] < landmarks[pip_index][1]


def _thumb_points_outward(landmarks, hand_label):
    """Check whether the thumb points away from the palm on the x-axis."""
    thumb_tip_x = landmarks[4][0]
    thumb_ip_x = landmarks[3][0]
    if hand_label == "Left":
        return thumb_tip_x > thumb_ip_x
    return thumb_tip_x < thumb_ip_x


def _thumb_points_up(landmarks):
    """Check whether the thumb direction is mainly upward."""
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    vertical_change = thumb_mcp[1] - thumb_tip[1]
    horizontal_change = abs(thumb_tip[0] - thumb_mcp[0])
    return vertical_change > max(12, horizontal_change * 0.65)


def _thumb_points_down(landmarks):
    """Check whether the thumb direction is mainly downward."""
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    vertical_change = thumb_tip[1] - thumb_mcp[1]
    horizontal_change = abs(thumb_tip[0] - thumb_mcp[0])
    return vertical_change > max(12, horizontal_change * 0.65)


class GestureRecognizer:
    """Classify one hand as one of the eight project gesture labels."""

    def recognize(self, landmark_list, hand_label="Right"):
        """Return the gesture name, rule confidence, and suspicious flag."""
        if len(landmark_list) != 21:
            return "Unknown Gesture", 0.0, False

        index_extended = _finger_is_extended(landmark_list, 8, 6)
        middle_extended = _finger_is_extended(landmark_list, 12, 10)
        ring_extended = _finger_is_extended(landmark_list, 16, 14)
        pinky_extended = _finger_is_extended(landmark_list, 20, 18)
        thumb_outward = _thumb_points_outward(landmark_list, hand_label)
        thumb_up = _thumb_points_up(landmark_list)
        thumb_down = _thumb_points_down(landmark_list)

        finger_states = [
            index_extended,
            middle_extended,
            ring_extended,
            pinky_extended,
        ]
        extended_finger_count = sum(finger_states)

        gesture_name = "Unknown Gesture"
        confidence = 0.35

        # More specific gestures are checked before general gestures.
        if thumb_outward and all(finger_states):
            gesture_name, confidence = "Open Palm", 0.92
        elif (
            thumb_outward
            and pinky_extended
            and not index_extended
            and not middle_extended
            and not ring_extended
        ):
            gesture_name, confidence = "Phone Gesture", 0.86
        elif (
            index_extended
            and middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            gesture_name, confidence = "Victory Sign", 0.88
        elif (
            index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            gesture_name, confidence = "Pointing Finger", 0.88
        elif thumb_up and extended_finger_count == 0:
            gesture_name, confidence = "Thumbs Up", 0.87
        elif thumb_down and extended_finger_count == 0:
            gesture_name, confidence = "Thumbs Down", 0.84
        elif extended_finger_count == 0 and not thumb_outward:
            gesture_name, confidence = "Closed Fist", 0.90

        if gesture_name not in GESTURE_LABELS:
            gesture_name = "Unknown Gesture"
            confidence = 0.0

        is_suspicious = gesture_name in SUSPICIOUS_GESTURES
        return gesture_name, confidence, is_suspicious


class MultiHandGestureRecognizer:
    """Apply GestureRecognizer to every hand detected in one frame."""

    def __init__(self):
        self.recognizer = GestureRecognizer()

    def recognize_all(self, hand_landmark_lists, hand_labels):
        """Return one simple recognition dictionary for each visible hand."""
        recognition_results = []
        for hand_index, landmark_list in enumerate(hand_landmark_lists):
            hand_label = (
                hand_labels[hand_index]
                if hand_index < len(hand_labels)
                else f"Hand {hand_index + 1}"
            )
            gesture_name, confidence, is_suspicious = self.recognizer.recognize(
                landmark_list, hand_label
            )
            recognition_results.append(
                {
                    "hand": hand_label,
                    "gesture": gesture_name,
                    "confidence": round(confidence, 2),
                    "is_suspicious": is_suspicious,
                }
            )
        return recognition_results

    def any_suspicious(self, recognition_results):
        """Return True when any visible hand shows a suspicious gesture."""
        return any(result["is_suspicious"] for result in recognition_results)
