"""Detect faces, landmarks, head direction, and frame movement.

This module uses the current MediaPipe Tasks API and local model files. Camera
frames are never sent to a remote service.
"""

import math
import time

import cv2

from config import FACE_DETECTOR_MODEL, FACE_LANDMARKER_MODEL
from mediapipe.tasks import python as media_pipe_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import Image as MediaPipeImage
from mediapipe.tasks.python.vision.core.image import ImageFormat


class FaceDetector:
    """Use MediaPipe Tasks to monitor all visible faces in a camera frame."""

    def __init__(self, settings):
        self.settings = settings
        self._check_model_files()

        detector_options = vision.FaceDetectorOptions(
            base_options=media_pipe_python.BaseOptions(
                model_asset_path=str(FACE_DETECTOR_MODEL)
            ),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=settings.face_detection_confidence,
        )
        landmarker_options = vision.FaceLandmarkerOptions(
            base_options=media_pipe_python.BaseOptions(
                model_asset_path=str(FACE_LANDMARKER_MODEL)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=5,
            min_face_detection_confidence=settings.face_detection_confidence,
            min_face_presence_confidence=settings.face_detection_confidence,
            min_tracking_confidence=settings.face_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.face_detector = vision.FaceDetector.create_from_options(detector_options)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(
            landmarker_options
        )
        self.face_count = 0
        self.face_bboxes = []
        self.last_results = self._empty_results()
        self._previous_face_center = None
        self._last_timestamp_ms = 0

    def _check_model_files(self):
        """Stop with a clear message when required model files are missing."""
        missing_models = [
            str(model_path)
            for model_path in (FACE_DETECTOR_MODEL, FACE_LANDMARKER_MODEL)
            if not model_path.exists()
        ]
        if missing_models:
            raise FileNotFoundError(
                "Missing MediaPipe face model files: " + ", ".join(missing_models)
            )

    def _empty_results(self):
        """Return a complete result object for frames with no visible face."""
        return {
            "face_count": 0,
            "face_bboxes": [],
            "face_confidences": [],
            "face_visible": False,
            "face_outside_frame": False,
            "is_looking_away": False,
            "look_away_ratio": 0.0,
            "face_center": None,
            "face_movement": 0.0,
        }

    def process_frame(self, frame):
        """Analyse one BGR frame and return the annotated frame and results."""
        if frame is None or frame.size == 0:
            raise ValueError("FaceDetector received an empty camera frame.")

        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        media_pipe_image = MediaPipeImage(
            image_format=ImageFormat.SRGB,
            data=rgb_frame,
        )
        timestamp_ms = self._next_timestamp_ms()
        detection_result = self.face_detector.detect_for_video(
            media_pipe_image, timestamp_ms
        )
        landmark_result = self.face_landmarker.detect_for_video(
            media_pipe_image, timestamp_ms
        )

        annotated_frame = frame.copy()
        face_bboxes = []
        face_confidences = []
        face_outside_frame = False

        for detection in detection_result.detections:
            bounding_box = detection.bounding_box
            raw_x = int(bounding_box.origin_x)
            raw_y = int(bounding_box.origin_y)
            raw_width = int(bounding_box.width)
            raw_height = int(bounding_box.height)

            if self._is_near_frame_edge(
                raw_x, raw_y, raw_width, raw_height, frame_width, frame_height
            ):
                face_outside_frame = True

            x = max(0, raw_x)
            y = max(0, raw_y)
            box_width = max(0, min(raw_width, frame_width - x))
            box_height = max(0, min(raw_height, frame_height - y))

            if box_width > 0 and box_height > 0:
                face_bboxes.append((x, y, box_width, box_height))
                confidence = (
                    float(detection.categories[0].score)
                    if detection.categories
                    else 0.0
                )
                face_confidences.append(confidence)

        self.face_count = len(face_bboxes)
        self.face_bboxes = face_bboxes

        face_landmark_groups = landmark_result.face_landmarks
        if self.settings.draw_face_mesh and face_landmark_groups:
            self._draw_face_contours(
                annotated_frame,
                face_landmark_groups,
                frame_width,
                frame_height,
            )

        look_away_ratio = 0.0
        is_looking_away = False
        if face_landmark_groups:
            look_away_ratio = self._calculate_look_away_ratio(
                face_landmark_groups[0], frame_width
            )
            is_looking_away = (
                look_away_ratio >= self.settings.look_away_ratio_threshold
            )

        face_center = self._get_primary_face_center(face_bboxes)
        face_movement = self._calculate_face_movement(face_center)
        self._draw_face_boxes(
            annotated_frame,
            face_bboxes,
            face_confidences,
            face_outside_frame,
        )

        self.last_results = {
            "face_count": self.face_count,
            "face_bboxes": face_bboxes,
            "face_confidences": face_confidences,
            "face_visible": self.face_count > 0,
            "face_outside_frame": face_outside_frame,
            "is_looking_away": is_looking_away,
            "look_away_ratio": round(look_away_ratio, 3),
            "face_center": face_center,
            "face_movement": round(face_movement, 2),
        }
        return annotated_frame, self.last_results.copy()

    def _next_timestamp_ms(self):
        """Return a strictly increasing timestamp required by VIDEO mode."""
        current_timestamp = int(time.monotonic() * 1000)
        self._last_timestamp_ms = max(current_timestamp, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def _is_near_frame_edge(self, x, y, width, height, frame_width, frame_height):
        """Check whether part of a face is touching or leaving the camera frame."""
        horizontal_margin = int(frame_width * self.settings.edge_margin_ratio)
        vertical_margin = int(frame_height * self.settings.edge_margin_ratio)
        return (
            x <= horizontal_margin
            or y <= vertical_margin
            or x + width >= frame_width - horizontal_margin
            or y + height >= frame_height - vertical_margin
        )

    def _calculate_look_away_ratio(self, face_landmarks, frame_width):
        """Estimate head turn from the nose position between both outer eyes."""
        nose_x = face_landmarks[1].x * frame_width
        left_eye_x = face_landmarks[33].x * frame_width
        right_eye_x = face_landmarks[263].x * frame_width
        eye_center_x = (left_eye_x + right_eye_x) / 2
        eye_distance = max(abs(right_eye_x - left_eye_x), 1.0)
        return abs(nose_x - eye_center_x) / eye_distance

    def _get_primary_face_center(self, face_bboxes):
        """Return the center point of the largest visible face."""
        if not face_bboxes:
            return None

        primary_box = max(face_bboxes, key=lambda box: box[2] * box[3])
        x, y, width, height = primary_box
        return (x + width // 2, y + height // 2)

    def _calculate_face_movement(self, face_center):
        """Measure how far the main face moved since the previous frame."""
        if face_center is None:
            self._previous_face_center = None
            return 0.0

        movement = 0.0
        if self._previous_face_center is not None:
            movement = math.dist(face_center, self._previous_face_center)
        self._previous_face_center = face_center
        return movement

    def _draw_face_contours(
        self, frame, face_landmark_groups, frame_width, frame_height
    ):
        """Draw face contour connections and landmark points."""
        connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
        for face_landmarks in face_landmark_groups:
            for connection in connections:
                start = face_landmarks[connection.start]
                end = face_landmarks[connection.end]
                start_point = (
                    int(start.x * frame_width),
                    int(start.y * frame_height),
                )
                end_point = (
                    int(end.x * frame_width),
                    int(end.y * frame_height),
                )
                cv2.line(frame, start_point, end_point, (80, 180, 255), 1)

    def _draw_face_boxes(self, frame, boxes, confidences, is_outside):
        """Draw face boxes with a separate confidence value for each face."""
        for index, box in enumerate(boxes):
            x, y, width, height = box
            confidence = confidences[index] if index < len(confidences) else 0.0
            color = (0, 0, 255) if len(boxes) > 1 or is_outside else (0, 200, 0)
            label = f"Face {index + 1}: {confidence:.0%}"
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(
                frame,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    def release(self):
        """Release the native MediaPipe Tasks resources."""
        self.face_detector.close()
        self.face_landmarker.close()
