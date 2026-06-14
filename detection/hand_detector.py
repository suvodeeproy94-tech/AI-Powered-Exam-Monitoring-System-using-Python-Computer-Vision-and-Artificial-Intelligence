"""Detect hands, draw landmarks, and measure hand movement.

The current MediaPipe Hand Landmarker Tasks API runs with a local model file.
Only simple landmark and movement values are returned to the behavior monitor.
"""

from collections import deque
import math
import time

import cv2

from config import HAND_LANDMARKER_MODEL
from mediapipe.tasks import python as media_pipe_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import Image as MediaPipeImage
from mediapipe.tasks.python.vision.core.image import ImageFormat


class HandDetector:
    """Use MediaPipe Tasks to detect up to two hands in a webcam frame."""

    def __init__(self, settings):
        self.settings = settings
        if not HAND_LANDMARKER_MODEL.exists():
            raise FileNotFoundError(
                f"Missing MediaPipe hand model file: {HAND_LANDMARKER_MODEL}"
            )

        landmarker_options = vision.HandLandmarkerOptions(
            base_options=media_pipe_python.BaseOptions(
                model_asset_path=str(HAND_LANDMARKER_MODEL)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=settings.max_hands,
            min_hand_detection_confidence=settings.hand_detection_confidence,
            min_hand_presence_confidence=settings.hand_detection_confidence,
            min_tracking_confidence=settings.hand_tracking_confidence,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(
            landmarker_options
        )

        self._previous_wrist_positions = {}
        self._movement_ratio_histories = {}
        self._last_timestamp_ms = 0
        self.hand_count = 0
        self.hand_landmarks = []
        self.hand_landmark_points = []
        self.hand_bboxes = []
        self.hand_labels = []
        self.movement_speeds = []
        self.movement_ratios = []
        self.excessive_movement = False
        self.last_results = self._empty_results()

    def _empty_results(self):
        """Return a complete result object for frames with no visible hand."""
        return {
            "hand_count": 0,
            "hand_bboxes": [],
            "hand_labels": [],
            "hand_landmarks": [],
            "hand_landmark_points": [],
            "movement_speeds": [],
            "movement_ratios": [],
            "excessive_movement": False,
        }

    def process_frame(self, frame, drawing_frame=None):
        """Detect hands and draw them on an optional already-annotated frame."""
        if frame is None or frame.size == 0:
            raise ValueError("HandDetector received an empty camera frame.")

        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        media_pipe_image = MediaPipeImage(
            image_format=ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self.hand_landmarker.detect_for_video(
            media_pipe_image, self._next_timestamp_ms()
        )
        annotated_frame = (
            drawing_frame.copy() if drawing_frame is not None else frame.copy()
        )

        self.hand_landmarks = list(result.hand_landmarks)
        self.hand_landmark_points = []
        self.hand_bboxes = []
        self.hand_labels = []
        self.movement_speeds = []
        self.movement_ratios = []
        self.excessive_movement = False
        current_wrist_positions = {}
        visible_tracking_keys = set()

        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
            hand_label = self._get_hand_label(result.handedness, hand_index)
            tracking_key = f"{hand_label}_{hand_index}"
            visible_tracking_keys.add(tracking_key)
            self.hand_labels.append(hand_label)

            pixel_points = self._get_pixel_points(
                hand_landmarks, frame_width, frame_height
            )
            self.hand_landmark_points.append(pixel_points)

            hand_box = self._get_hand_box(
                hand_landmarks, frame_width, frame_height
            )
            self.hand_bboxes.append(hand_box)
            self._draw_hand_landmarks(annotated_frame, pixel_points)

            wrist_position = self._get_wrist_position(
                hand_landmarks, frame_width, frame_height
            )
            current_wrist_positions[tracking_key] = wrist_position
            movement_speed = self._get_movement_speed(
                tracking_key, wrist_position
            )
            movement_ratio = self._get_smoothed_movement_ratio(
                tracking_key, movement_speed, hand_box
            )
            self.movement_speeds.append(movement_speed)
            self.movement_ratios.append(movement_ratio)

            if movement_ratio >= self.settings.hand_movement_ratio_threshold:
                self.excessive_movement = True

        self._previous_wrist_positions = current_wrist_positions
        self._remove_missing_movement_histories(visible_tracking_keys)
        self.hand_count = len(self.hand_landmarks)
        self.last_results = {
            "hand_count": self.hand_count,
            "hand_bboxes": list(self.hand_bboxes),
            "hand_labels": list(self.hand_labels),
            "hand_landmarks": list(self.hand_landmarks),
            "hand_landmark_points": list(self.hand_landmark_points),
            "movement_speeds": [round(speed, 2) for speed in self.movement_speeds],
            "movement_ratios": [round(ratio, 3) for ratio in self.movement_ratios],
            "excessive_movement": self.excessive_movement,
        }
        return annotated_frame, self.last_results.copy()

    def _next_timestamp_ms(self):
        """Return a strictly increasing timestamp required by VIDEO mode."""
        current_timestamp = int(time.monotonic() * 1000)
        self._last_timestamp_ms = max(current_timestamp, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def _get_hand_label(self, handedness_groups, hand_index):
        """Read the Left or Right label supplied by MediaPipe."""
        if hand_index >= len(handedness_groups) or not handedness_groups[hand_index]:
            return f"Hand {hand_index + 1}"
        return handedness_groups[hand_index][0].category_name or f"Hand {hand_index + 1}"

    def _get_hand_box(self, hand_landmarks, frame_width, frame_height):
        """Create a padded pixel bounding box around all 21 hand landmarks."""
        x_points = [landmark.x * frame_width for landmark in hand_landmarks]
        y_points = [landmark.y * frame_height for landmark in hand_landmarks]
        padding = 10

        x_min = max(0, int(min(x_points)) - padding)
        y_min = max(0, int(min(y_points)) - padding)
        x_max = min(frame_width, int(max(x_points)) + padding)
        y_max = min(frame_height, int(max(y_points)) + padding)
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def _get_wrist_position(self, hand_landmarks, frame_width, frame_height):
        """Convert the wrist landmark into a pixel position."""
        wrist = hand_landmarks[0]
        return (int(wrist.x * frame_width), int(wrist.y * frame_height))

    def _get_movement_speed(self, tracking_key, wrist_position):
        """Measure wrist travel since the previous frame."""
        previous_position = self._previous_wrist_positions.get(tracking_key)
        if previous_position is None:
            return 0.0
        return math.dist(wrist_position, previous_position)

    def _get_smoothed_movement_ratio(self, tracking_key, movement, hand_box):
        """Compare movement with hand size and smooth small landmark jumps."""
        _, _, box_width, box_height = hand_box
        hand_size = max(math.hypot(box_width, box_height), 1.0)
        movement_ratio = movement / hand_size

        history = self._movement_ratio_histories.setdefault(
            tracking_key, deque(maxlen=3)
        )
        history.append(movement_ratio)
        return sum(history) / len(history)

    def _remove_missing_movement_histories(self, visible_tracking_keys):
        """Forget movement history after a hand leaves the camera frame."""
        missing_keys = set(self._movement_ratio_histories) - visible_tracking_keys
        for tracking_key in missing_keys:
            self._movement_ratio_histories.pop(tracking_key, None)

    def _get_pixel_points(self, hand_landmarks, frame_width, frame_height):
        """Convert all hand landmarks into reusable pixel points."""
        return [
            (int(point.x * frame_width), int(point.y * frame_height))
            for point in hand_landmarks
        ]

    def _draw_hand_landmarks(self, frame, pixel_points):
        """Draw white hand bones with clear red landmark points."""
        for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(
                frame,
                pixel_points[connection.start],
                pixel_points[connection.end],
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
        for point in pixel_points:
            cv2.circle(frame, point, 5, (245, 245, 245), -1, cv2.LINE_AA)
            cv2.circle(frame, point, 3, (0, 0, 255), -1, cv2.LINE_AA)

    def get_landmark_list(self, hand_index=0, frame_shape=(480, 640)):
        """Return all 21 landmarks as simple pixel coordinate tuples."""
        if hand_index < 0 or hand_index >= len(self.hand_landmarks):
            return []

        frame_height, frame_width = frame_shape[:2]
        selected_hand = self.hand_landmarks[hand_index]
        return [
            (int(point.x * frame_width), int(point.y * frame_height))
            for point in selected_hand
        ]

    def release(self):
        """Release the native MediaPipe Tasks resources."""
        self.hand_landmarker.close()
