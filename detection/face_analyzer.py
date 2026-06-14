"""Estimate head pose, eye state, and gaze direction from face landmarks.

This module receives MediaPipe face landmarks from the face detector. It keeps
the mathematics separate from camera access so the calculations can be tested
without opening a webcam.
"""

from collections import deque
import math

import cv2
import numpy as np


HEAD_POSE_INDEXES = (1, 152, 33, 263, 61, 291)
HEAD_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


class FaceAnalyzer:
    """Calculate stable head and eye measurements for the primary face."""

    def __init__(self, settings):
        self.settings = settings
        self.calibration_profile = {}
        self._history = deque(maxlen=5)

    def set_calibration_profile(self, profile):
        """Apply neutral head and gaze values measured before monitoring."""
        self.calibration_profile = dict(profile or {})
        self._history.clear()

    def analyse(self, face_landmarks, frame_width, frame_height):
        """Return head pose, gaze, eye state, and final attention direction."""
        if not face_landmarks or len(face_landmarks) < 292:
            return self.empty_results()

        head_pose = self._calculate_head_pose(
            face_landmarks, frame_width, frame_height
        )
        eye_results = self._calculate_eye_results(
            face_landmarks, frame_width, frame_height
        )

        raw_results = {**head_pose, **eye_results}
        stable_results = self._smooth_results(raw_results)
        return self._apply_calibration_and_thresholds(stable_results)

    def empty_results(self):
        """Return a complete result when detailed face landmarks are missing."""
        return {
            "head_pose_available": False,
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "head_roll": 0.0,
            "gaze_available": False,
            "gaze_horizontal": 0.0,
            "gaze_vertical": 0.0,
            "gaze_direction": "Unknown",
            "left_eye_ratio": 0.0,
            "right_eye_ratio": 0.0,
            "eyes_closed": False,
            "is_looking_away": False,
            "attention_reason": "Face landmarks unavailable",
        }

    def _calculate_head_pose(self, landmarks, frame_width, frame_height):
        """Use six face points and solvePnP to estimate head rotation."""
        image_points = np.array(
            [
                (
                    landmarks[index].x * frame_width,
                    landmarks[index].y * frame_height,
                )
                for index in HEAD_POSE_INDEXES
            ],
            dtype=np.float64,
        )
        focal_length = float(frame_width)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, frame_width / 2],
                [0.0, focal_length, frame_height / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rotation_vector, _ = cv2.solvePnP(
                HEAD_MODEL_POINTS,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            success = False
        if not success:
            return {
                "head_pose_available": False,
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "head_roll": 0.0,
            }

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = cv2.RQDecomp3x3(rotation_matrix)[0]
        pitch = self._normalize_pitch(pitch)
        return {
            "head_pose_available": True,
            "head_yaw": float(yaw),
            "head_pitch": float(pitch),
            "head_roll": float(roll),
        }

    def _calculate_eye_results(self, landmarks, frame_width, frame_height):
        """Measure eye opening and iris position inside both eyes."""
        left_eye_ratio = self._eye_aspect_ratio(
            landmarks, (33, 133, 159, 145), frame_width, frame_height
        )
        right_eye_ratio = self._eye_aspect_ratio(
            landmarks, (362, 263, 386, 374), frame_width, frame_height
        )
        average_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        results = {
            "left_eye_ratio": left_eye_ratio,
            "right_eye_ratio": right_eye_ratio,
            "eyes_closed": average_eye_ratio < self.settings.eye_closed_threshold,
            "gaze_available": len(landmarks) >= 478,
            "gaze_horizontal": 0.0,
            "gaze_vertical": 0.0,
        }

        if len(landmarks) >= 478:
            left_gaze = self._iris_position(
                landmarks, 468, (33, 133, 159, 145), frame_width, frame_height
            )
            right_gaze = self._iris_position(
                landmarks, 473, (362, 263, 386, 374), frame_width, frame_height
            )
            results["gaze_horizontal"] = (left_gaze[0] + right_gaze[0]) / 2
            results["gaze_vertical"] = (left_gaze[1] + right_gaze[1]) / 2

        return results

    def _eye_aspect_ratio(
        self, landmarks, indexes, frame_width, frame_height
    ):
        """Return vertical eye opening divided by horizontal eye width."""
        left_index, right_index, top_index, bottom_index = indexes
        left_point = self._point(landmarks[left_index], frame_width, frame_height)
        right_point = self._point(landmarks[right_index], frame_width, frame_height)
        top_point = self._point(landmarks[top_index], frame_width, frame_height)
        bottom_point = self._point(
            landmarks[bottom_index], frame_width, frame_height
        )
        horizontal_distance = max(math.dist(left_point, right_point), 1.0)
        return math.dist(top_point, bottom_point) / horizontal_distance

    def _iris_position(
        self, landmarks, iris_index, eye_indexes, frame_width, frame_height
    ):
        """Return iris displacement from the eye center in a -1 to 1 range."""
        iris_x, iris_y = self._point(
            landmarks[iris_index], frame_width, frame_height
        )
        eye_points = [
            self._point(landmarks[index], frame_width, frame_height)
            for index in eye_indexes
        ]
        minimum_x = min(point[0] for point in eye_points)
        maximum_x = max(point[0] for point in eye_points)
        minimum_y = min(point[1] for point in eye_points)
        maximum_y = max(point[1] for point in eye_points)
        horizontal = ((iris_x - minimum_x) / max(maximum_x - minimum_x, 1.0) - 0.5) * 2
        vertical = ((iris_y - minimum_y) / max(maximum_y - minimum_y, 1.0) - 0.5) * 2
        return self._clamp(horizontal), self._clamp(vertical)

    def _smooth_results(self, results):
        """Average recent numeric measurements to reduce landmark shaking."""
        self._history.append(results)
        numeric_names = (
            "head_yaw",
            "head_pitch",
            "head_roll",
            "gaze_horizontal",
            "gaze_vertical",
            "left_eye_ratio",
            "right_eye_ratio",
        )
        stable_results = dict(results)
        for name in numeric_names:
            values = [float(item.get(name, 0.0)) for item in self._history]
            stable_results[name] = sum(values) / len(values)
        stable_results["eyes_closed"] = sum(
            bool(item.get("eyes_closed")) for item in self._history
        ) >= math.ceil(len(self._history) * 0.6)
        return stable_results

    def _apply_calibration_and_thresholds(self, results):
        """Compare stable values with the student's neutral calibration pose."""
        yaw = results["head_yaw"] - self.calibration_profile.get("head_yaw", 0.0)
        pitch = results["head_pitch"] - self.calibration_profile.get(
            "head_pitch", 0.0
        )
        gaze_horizontal = results["gaze_horizontal"] - self.calibration_profile.get(
            "gaze_horizontal", 0.0
        )
        gaze_vertical = results["gaze_vertical"] - self.calibration_profile.get(
            "gaze_vertical", 0.0
        )

        head_away = self.settings.head_pose_enabled and (
            abs(yaw) >= self.settings.head_yaw_threshold
            or abs(pitch) >= self.settings.head_pitch_threshold
        )
        gaze_away = self.settings.gaze_tracking_enabled and results[
            "gaze_available"
        ] and (
            abs(gaze_horizontal) >= self.settings.gaze_horizontal_threshold
            or abs(gaze_vertical) >= self.settings.gaze_vertical_threshold
        )

        gaze_direction = "Center"
        if gaze_horizontal <= -self.settings.gaze_horizontal_threshold:
            gaze_direction = "Left"
        elif gaze_horizontal >= self.settings.gaze_horizontal_threshold:
            gaze_direction = "Right"
        elif gaze_vertical <= -self.settings.gaze_vertical_threshold:
            gaze_direction = "Up"
        elif gaze_vertical >= self.settings.gaze_vertical_threshold:
            gaze_direction = "Down"

        attention_reason = "Looking at screen"
        if results["eyes_closed"]:
            attention_reason = "Eyes closed"
        elif head_away:
            attention_reason = "Head turned away"
        elif gaze_away:
            attention_reason = f"Eyes looking {gaze_direction.lower()}"

        results.update(
            {
                "head_yaw": round(yaw, 2),
                "head_pitch": round(pitch, 2),
                "head_roll": round(results["head_roll"], 2),
                "gaze_horizontal": round(gaze_horizontal, 3),
                "gaze_vertical": round(gaze_vertical, 3),
                "gaze_direction": gaze_direction,
                "left_eye_ratio": round(results["left_eye_ratio"], 3),
                "right_eye_ratio": round(results["right_eye_ratio"], 3),
                "is_looking_away": head_away or gaze_away,
                "attention_reason": attention_reason,
            }
        )
        return results

    def _point(self, landmark, frame_width, frame_height):
        """Convert one normalized MediaPipe landmark to pixel coordinates."""
        return landmark.x * frame_width, landmark.y * frame_height

    def _clamp(self, value):
        """Keep a gaze value inside the expected -1 to 1 range."""
        return max(-1.0, min(1.0, float(value)))

    def _normalize_pitch(self, pitch):
        """Convert OpenCV's rear-facing pitch form into an intuitive angle."""
        normalized_pitch = float(pitch)
        if normalized_pitch < -90.0:
            normalized_pitch = -180.0 - normalized_pitch
        elif normalized_pitch > 90.0:
            normalized_pitch = 180.0 - normalized_pitch
        return normalized_pitch
