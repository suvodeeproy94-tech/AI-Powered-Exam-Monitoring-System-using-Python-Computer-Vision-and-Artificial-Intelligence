"""Recognise hand gestures with rotation-independent landmark geometry.

Finger state is calculated from joint angles and palm-relative distances. This
works when a hand is vertical, diagonal, or slightly rotated, unlike a simple
screen y-coordinate comparison. A short voting history removes one-frame label
changes caused by motion blur.
"""

from collections import Counter, deque
import math

from config import AppSettings, GESTURE_LABELS, SUSPICIOUS_GESTURES
from recognition.trained_gesture_model import TrainedGestureModel


FINGER_JOINTS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _xy(point):
    """Return the x and y values from a two- or three-value landmark."""
    return float(point[0]), float(point[1])


def _distance(first_point, second_point):
    """Calculate straight-line distance between two landmark points."""
    first_x, first_y = _xy(first_point)
    second_x, second_y = _xy(second_point)
    return math.hypot(first_x - second_x, first_y - second_y)


def _angle(first_point, middle_point, last_point):
    """Return the joint angle at the middle point in degrees."""
    first_x, first_y = _xy(first_point)
    middle_x, middle_y = _xy(middle_point)
    last_x, last_y = _xy(last_point)

    first_vector = (first_x - middle_x, first_y - middle_y)
    second_vector = (last_x - middle_x, last_y - middle_y)
    first_length = math.hypot(*first_vector)
    second_length = math.hypot(*second_vector)
    if first_length == 0 or second_length == 0:
        return 0.0

    cosine_value = (
        first_vector[0] * second_vector[0]
        + first_vector[1] * second_vector[1]
    ) / (first_length * second_length)
    cosine_value = max(-1.0, min(1.0, cosine_value))
    return math.degrees(math.acos(cosine_value))


def _clamp(value, minimum=0.0, maximum=1.0):
    """Keep a numeric score inside a fixed range."""
    return max(minimum, min(maximum, value))


def _finger_extension_score(landmarks, joint_indexes):
    """Return a 0 to 1 score showing how straight and extended a finger is."""
    mcp_index, pip_index, dip_index, tip_index = joint_indexes
    pip_angle = _angle(
        landmarks[mcp_index], landmarks[pip_index], landmarks[dip_index]
    )
    dip_angle = _angle(
        landmarks[pip_index], landmarks[dip_index], landmarks[tip_index]
    )
    straightness = _clamp((min(pip_angle, dip_angle) - 95.0) / 70.0)

    wrist_to_pip = max(_distance(landmarks[0], landmarks[pip_index]), 0.0001)
    wrist_to_tip = _distance(landmarks[0], landmarks[tip_index])
    reach_ratio = wrist_to_tip / wrist_to_pip
    reach_score = _clamp((reach_ratio - 1.0) / 0.35)
    return 0.75 * straightness + 0.25 * reach_score


def _thumb_extension_score(landmarks):
    """Return a 0 to 1 score showing whether the thumb is straight and open."""
    mcp_angle = _angle(landmarks[1], landmarks[2], landmarks[3])
    ip_angle = _angle(landmarks[2], landmarks[3], landmarks[4])
    straightness = _clamp((min(mcp_angle, ip_angle) - 90.0) / 75.0)

    palm_center = (
        sum(_xy(landmarks[index])[0] for index in (0, 5, 9, 13, 17)) / 5,
        sum(_xy(landmarks[index])[1] for index in (0, 5, 9, 13, 17)) / 5,
    )
    palm_to_ip = max(_distance(palm_center, landmarks[3]), 0.0001)
    palm_to_tip = _distance(palm_center, landmarks[4])
    reach_score = _clamp((palm_to_tip / palm_to_ip - 1.0) / 0.45)
    return 0.70 * straightness + 0.30 * reach_score


def _thumb_vertical_direction(landmarks):
    """Return up, down, or sideways from the thumb MCP-to-tip direction."""
    thumb_mcp_x, thumb_mcp_y = _xy(landmarks[2])
    thumb_tip_x, thumb_tip_y = _xy(landmarks[4])
    horizontal_change = thumb_tip_x - thumb_mcp_x
    vertical_change = thumb_tip_y - thumb_mcp_y

    if abs(vertical_change) < abs(horizontal_change) * 0.75:
        return "sideways"
    return "up" if vertical_change < 0 else "down"


def _expected_state_quality(extension_scores, expected_extended):
    """Measure how closely finger scores match one expected pose."""
    qualities = []
    for finger_name, should_be_extended in expected_extended.items():
        score = extension_scores[finger_name]
        qualities.append(score if should_be_extended else 1.0 - score)
    return sum(qualities) / len(qualities)


class GestureRecognizer:
    """Classify one hand as one of the required exam-monitoring gestures."""

    def __init__(self, minimum_confidence=0.58, settings=None):
        self.minimum_confidence = minimum_confidence
        self.settings = settings or AppSettings()
        self.trained_model = TrainedGestureModel()

    def recognize(self, landmark_list, hand_label="Right"):
        """Return a gesture name, geometric confidence, and suspicious flag."""
        del hand_label
        if len(landmark_list) != 21:
            return "Unknown Gesture", 0.0, False

        if self.settings.trained_gesture_model_enabled:
            model_gesture, model_confidence = self.trained_model.predict(
                landmark_list
            )
            if model_confidence >= self.settings.trained_gesture_min_confidence:
                return (
                    model_gesture,
                    round(model_confidence, 2),
                    model_gesture in SUSPICIOUS_GESTURES,
                )

        extension_scores = {
            finger_name: _finger_extension_score(landmark_list, joint_indexes)
            for finger_name, joint_indexes in FINGER_JOINTS.items()
        }
        thumb_score = _thumb_extension_score(landmark_list)
        thumb_direction = _thumb_vertical_direction(landmark_list)
        extended = {
            finger_name: score >= 0.58
            for finger_name, score in extension_scores.items()
        }

        gesture_name = "Unknown Gesture"
        quality = 0.0

        # Phone is checked first because it uses an open thumb and pinky.
        phone_pattern = {
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": True,
        }
        if (
            thumb_score >= 0.58
            and extended["pinky"]
            and not extended["index"]
            and not extended["middle"]
            and not extended["ring"]
        ):
            gesture_name = "Phone Gesture"
            quality = 0.75 * _expected_state_quality(
                extension_scores, phone_pattern
            ) + 0.25 * thumb_score

        # Four straight fingers are an open palm even when the hand is rotated.
        elif all(extended.values()):
            gesture_name = "Open Palm"
            quality = _expected_state_quality(
                extension_scores,
                {finger_name: True for finger_name in FINGER_JOINTS},
            )

        elif (
            extended["index"]
            and extended["middle"]
            and not extended["ring"]
            and not extended["pinky"]
        ):
            gesture_name = "Victory Sign"
            quality = _expected_state_quality(
                extension_scores,
                {"index": True, "middle": True, "ring": False, "pinky": False},
            )

        elif (
            extended["index"]
            and not extended["middle"]
            and not extended["ring"]
            and not extended["pinky"]
        ):
            gesture_name = "Pointing Finger"
            quality = _expected_state_quality(
                extension_scores,
                {"index": True, "middle": False, "ring": False, "pinky": False},
            )

        elif not any(extended.values()) and thumb_score >= 0.58:
            folded_quality = _expected_state_quality(
                extension_scores,
                {finger_name: False for finger_name in FINGER_JOINTS},
            )
            quality = 0.70 * folded_quality + 0.30 * thumb_score
            if thumb_direction == "up":
                gesture_name = "Thumbs Up"
            elif thumb_direction == "down":
                gesture_name = "Thumbs Down"

        elif not any(extended.values()) and thumb_score < 0.58:
            gesture_name = "Closed Fist"
            quality = 0.80 * _expected_state_quality(
                extension_scores,
                {finger_name: False for finger_name in FINGER_JOINTS},
            ) + 0.20 * (1.0 - thumb_score)

        confidence = round(_clamp(quality), 2)
        if confidence < self.minimum_confidence or gesture_name not in GESTURE_LABELS:
            gesture_name = "Unknown Gesture"
            confidence = 0.0

        is_suspicious = gesture_name in SUSPICIOUS_GESTURES
        return gesture_name, confidence, is_suspicious


class MultiHandGestureRecognizer:
    """Recognise all hands and keep only stable multi-frame gesture results."""

    def __init__(self, settings=None):
        self.settings = settings or AppSettings()
        self.recognizer = GestureRecognizer(
            self.settings.gesture_min_confidence,
            self.settings,
        )
        self._gesture_histories = {}
        self._stable_results = {}

    def recognize_all(self, hand_landmark_lists, hand_labels):
        """Return temporally stable recognition results for visible hands."""
        recognition_results = []
        visible_keys = set()

        for hand_index, landmark_list in enumerate(hand_landmark_lists):
            hand_label = (
                hand_labels[hand_index]
                if hand_index < len(hand_labels)
                else f"Hand {hand_index + 1}"
            )
            tracking_key = f"{hand_label}:{hand_index}"
            visible_keys.add(tracking_key)
            raw_gesture, raw_confidence, _ = self.recognizer.recognize(
                landmark_list, hand_label
            )

            history = self._gesture_histories.setdefault(
                tracking_key,
                deque(maxlen=self.settings.gesture_history_frames),
            )
            history.append((raw_gesture, raw_confidence))
            stable_gesture, stable_confidence = self._get_stable_gesture(
                tracking_key, history
            )

            recognition_results.append(
                {
                    "hand": hand_label,
                    "gesture": stable_gesture,
                    "confidence": stable_confidence,
                    "is_suspicious": stable_gesture in SUSPICIOUS_GESTURES,
                    "raw_gesture": raw_gesture,
                    "raw_confidence": raw_confidence,
                }
            )

        self._remove_missing_hands(visible_keys)
        return recognition_results

    def _get_stable_gesture(self, tracking_key, history):
        """Use majority voting before changing the displayed gesture."""
        gesture_counts = Counter(gesture for gesture, _ in history)
        candidate_gesture, candidate_count = gesture_counts.most_common(1)[0]
        majority_frames = math.ceil(
            len(history) * self.settings.gesture_majority_ratio
        )
        required_frames = max(
            self.settings.gesture_stable_frames,
            majority_frames,
        )

        if candidate_count >= required_frames:
            matching_confidences = [
                confidence
                for gesture, confidence in history
                if gesture == candidate_gesture
            ]
            stable_result = (
                candidate_gesture,
                round(sum(matching_confidences) / len(matching_confidences), 2),
            )
            self._stable_results[tracking_key] = stable_result

        return self._stable_results.get(tracking_key, ("Unknown Gesture", 0.0))

    def _remove_missing_hands(self, visible_keys):
        """Forget history immediately after a hand disappears."""
        missing_keys = set(self._gesture_histories) - visible_keys
        for tracking_key in missing_keys:
            self._gesture_histories.pop(tracking_key, None)
            self._stable_results.pop(tracking_key, None)

    def reset(self):
        """Clear all gesture history before a new monitoring session."""
        self._gesture_histories.clear()
        self._stable_results.clear()

    def any_suspicious(self, recognition_results):
        """Return True when any stable result is a suspicious gesture."""
        return any(result["is_suspicious"] for result in recognition_results)
