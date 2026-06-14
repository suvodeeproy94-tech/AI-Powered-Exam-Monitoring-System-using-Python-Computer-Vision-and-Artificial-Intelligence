"""Detect faces, landmarks, head direction, and frame movement.

This module uses the current MediaPipe Tasks API and local model files. Camera
frames are never sent to a remote service.
"""

import math
import time

import cv2

from config import (
    FACE_DETECTOR_MODEL,
    FACE_LANDMARKER_MODEL,
    YUNET_FACE_DETECTOR_MODEL,
)
from detection.face_analyzer import FaceAnalyzer
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
        self.yunet_detector = self._create_yunet_detector()
        self._yunet_runtime_failed = False
        self.face_analyzer = FaceAnalyzer(settings)
        self.face_count = 0
        self.face_bboxes = []
        self.last_results = self._empty_results()
        self._previous_face_center = None
        self._smoothed_face_center = None
        self._missed_face_frames = 0
        self._last_primary_face_box = None
        self._next_track_id = 1
        self._active_track_id = None
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

    def _create_yunet_detector(self):
        """Load YuNet when it is enabled and its local ONNX file exists."""
        if not self.settings.yunet_face_detection_enabled:
            return None
        if not YUNET_FACE_DETECTOR_MODEL.exists():
            return None

        return cv2.FaceDetectorYN.create(
            str(YUNET_FACE_DETECTOR_MODEL),
            "",
            (self.settings.camera_width, self.settings.camera_height),
            self.settings.yunet_score_threshold,
            self.settings.yunet_nms_threshold,
            self.settings.yunet_top_k,
        )

    def _empty_results(self):
        """Return a complete result object for frames with no visible face."""
        return {
            "face_count": 0,
            "raw_face_count": 0,
            "face_bboxes": [],
            "face_confidences": [],
            "face_visible": False,
            "face_outside_frame": False,
            "is_looking_away": False,
            "look_away_ratio": 0.0,
            "face_center": None,
            "face_movement": 0.0,
            "face_movement_ratio": 0.0,
            "stable_face_visible": False,
            "primary_face_area_ratio": 0.0,
            "primary_face_track_id": None,
            "face_detector_backend": "YuNet" if self.yunet_detector else "MediaPipe",
            **self.face_analyzer.empty_results(),
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
        landmark_result = self.face_landmarker.detect_for_video(
            media_pipe_image, timestamp_ms
        )

        annotated_frame = frame.copy()
        (
            face_bboxes,
            face_confidences,
            face_outside_frame,
            detector_backend,
        ) = self._detect_faces(
            frame,
            media_pipe_image,
            timestamp_ms,
            frame_width,
            frame_height,
        )

        raw_face_count = len(face_bboxes)
        self.face_count = raw_face_count
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
        legacy_looking_away = False
        if face_landmark_groups:
            look_away_ratio = self._calculate_look_away_ratio(
                face_landmark_groups[0], frame_width
            )
            legacy_looking_away = (
                look_away_ratio >= self.settings.look_away_ratio_threshold
            )

        detailed_analysis = self.face_analyzer.analyse(
            face_landmark_groups[0] if face_landmark_groups else None,
            frame_width,
            frame_height,
        )
        detailed_analysis["is_looking_away"] = (
            detailed_analysis.get("is_looking_away", False)
            or legacy_looking_away
        )

        face_center = self._get_primary_face_center(face_bboxes)
        primary_face_box = self._get_primary_face_box(face_bboxes)
        stable_face_visible, tracked_face_box, track_id = self._update_face_tracking(
            primary_face_box
        )
        face_movement, face_movement_ratio = self._calculate_face_movement(
            face_center, primary_face_box
        )
        primary_face_area_ratio = self._get_face_area_ratio(
            primary_face_box, frame_width, frame_height
        )
        self._draw_face_boxes(
            annotated_frame,
            face_bboxes,
            face_confidences,
            face_outside_frame,
        )

        self.last_results = {
            "face_count": self.face_count,
            "raw_face_count": raw_face_count,
            "face_bboxes": face_bboxes,
            "face_confidences": face_confidences,
            "face_visible": self.face_count > 0,
            "stable_face_visible": stable_face_visible,
            "tracked_face_box": tracked_face_box,
            "primary_face_track_id": track_id,
            "face_detector_backend": detector_backend,
            "face_outside_frame": face_outside_frame,
            "is_looking_away": detailed_analysis["is_looking_away"],
            "look_away_ratio": round(look_away_ratio, 3),
            "face_center": face_center,
            "face_movement": round(face_movement, 2),
            "face_movement_ratio": round(face_movement_ratio, 3),
            "primary_face_area_ratio": round(primary_face_area_ratio, 4),
            **detailed_analysis,
        }
        return annotated_frame, self.last_results.copy()

    def _detect_faces(
        self,
        frame,
        media_pipe_image,
        timestamp_ms,
        frame_width,
        frame_height,
    ):
        """Use YuNet first and automatically fall back to MediaPipe."""
        if self.yunet_detector is not None and not self._yunet_runtime_failed:
            try:
                return self._detect_faces_with_yunet(
                    frame, frame_width, frame_height
                )
            except (cv2.error, RuntimeError, ValueError):
                # A local OpenCV or model problem must not stop the exam session.
                self._yunet_runtime_failed = True

        return self._detect_faces_with_mediapipe(
            media_pipe_image,
            timestamp_ms,
            frame_width,
            frame_height,
        )

    def _detect_faces_with_yunet(self, frame, frame_width, frame_height):
        """Detect all visible faces with the local YuNet ONNX model."""
        self.yunet_detector.setInputSize((frame_width, frame_height))
        _, detected_faces = self.yunet_detector.detect(frame)
        raw_faces = [] if detected_faces is None else detected_faces

        face_bboxes = []
        face_confidences = []
        face_outside_frame = False
        for detected_face in raw_faces:
            raw_x = int(round(float(detected_face[0])))
            raw_y = int(round(float(detected_face[1])))
            raw_width = int(round(float(detected_face[2])))
            raw_height = int(round(float(detected_face[3])))
            confidence = float(detected_face[-1])
            face_outside_frame = self._add_face_candidate(
                face_bboxes,
                face_confidences,
                raw_x,
                raw_y,
                raw_width,
                raw_height,
                confidence,
                frame_width,
                frame_height,
            ) or face_outside_frame

        return face_bboxes, face_confidences, face_outside_frame, "YuNet"

    def _detect_faces_with_mediapipe(
        self, media_pipe_image, timestamp_ms, frame_width, frame_height
    ):
        """Use the original detector when YuNet is disabled or unavailable."""
        detection_result = self.face_detector.detect_for_video(
            media_pipe_image, timestamp_ms
        )
        face_bboxes = []
        face_confidences = []
        face_outside_frame = False

        for detection in detection_result.detections:
            bounding_box = detection.bounding_box
            confidence = (
                float(detection.categories[0].score)
                if detection.categories
                else 0.0
            )
            face_outside_frame = self._add_face_candidate(
                face_bboxes,
                face_confidences,
                int(bounding_box.origin_x),
                int(bounding_box.origin_y),
                int(bounding_box.width),
                int(bounding_box.height),
                confidence,
                frame_width,
                frame_height,
            ) or face_outside_frame

        return face_bboxes, face_confidences, face_outside_frame, "MediaPipe"

    def _add_face_candidate(
        self,
        face_bboxes,
        face_confidences,
        raw_x,
        raw_y,
        raw_width,
        raw_height,
        confidence,
        frame_width,
        frame_height,
    ):
        """Validate, clip, and add one detector result to the face list."""
        x = max(0, raw_x)
        y = max(0, raw_y)
        box_width = max(0, min(raw_width, frame_width - x))
        box_height = max(0, min(raw_height, frame_height - y))
        face_area_ratio = box_width * box_height / max(
            frame_width * frame_height, 1
        )
        if (
            box_width <= 0
            or box_height <= 0
            or face_area_ratio < self.settings.minimum_face_area_ratio
        ):
            return False

        self._append_distinct_face(
            face_bboxes,
            face_confidences,
            (x, y, box_width, box_height),
            confidence,
        )
        return self._is_near_frame_edge(
            raw_x,
            raw_y,
            raw_width,
            raw_height,
            frame_width,
            frame_height,
        )

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
        primary_box = self._get_primary_face_box(face_bboxes)
        if primary_box is None:
            return None

        x, y, width, height = primary_box
        return (x + width // 2, y + height // 2)

    def _get_primary_face_box(self, face_bboxes):
        """Return the largest face box because it is normally the student."""
        if not face_bboxes:
            return None
        return max(face_bboxes, key=lambda box: box[2] * box[3])

    def _update_face_tracking(self, primary_face_box):
        """Keep the primary face stable across a few missed detector frames."""
        if primary_face_box is not None:
            self._missed_face_frames = 0
            if self._last_primary_face_box is None:
                self._last_primary_face_box = primary_face_box
                self._active_track_id = self._next_track_id
                self._next_track_id += 1
            else:
                smoothing_weight = 0.45
                self._last_primary_face_box = tuple(
                    int(previous + (current - previous) * smoothing_weight)
                    for previous, current in zip(
                        self._last_primary_face_box, primary_face_box
                    )
                )
            return True, self._last_primary_face_box, self._active_track_id

        self._missed_face_frames += 1
        if self._missed_face_frames <= self.settings.face_tracking_grace_frames:
            return (
                self._last_primary_face_box is not None,
                self._last_primary_face_box,
                self._active_track_id,
            )
        self._last_primary_face_box = None
        self._active_track_id = None
        return False, None, None

    def _get_face_area_ratio(self, face_box, frame_width, frame_height):
        """Return how much of the camera frame is occupied by the main face."""
        if face_box is None:
            return 0.0
        _, _, face_width, face_height = face_box
        return face_width * face_height / max(frame_width * frame_height, 1)

    def set_calibration_profile(self, profile):
        """Apply the completed personal neutral pose to detailed face analysis."""
        self.face_analyzer.set_calibration_profile(profile)

    def _calculate_face_movement(self, face_center, face_box):
        """Measure stable face movement relative to the visible face size."""
        if face_center is None:
            self._previous_face_center = None
            self._smoothed_face_center = None
            return 0.0, 0.0

        if self._smoothed_face_center is None:
            self._smoothed_face_center = face_center
        else:
            previous_x, previous_y = self._smoothed_face_center
            current_x, current_y = face_center
            smoothing_weight = 0.35
            self._smoothed_face_center = (
                previous_x + (current_x - previous_x) * smoothing_weight,
                previous_y + (current_y - previous_y) * smoothing_weight,
            )

        movement = 0.0
        if self._previous_face_center is not None:
            movement = math.dist(
                self._smoothed_face_center, self._previous_face_center
            )
        self._previous_face_center = self._smoothed_face_center

        if face_box is None:
            return movement, 0.0
        face_size = max(math.hypot(face_box[2], face_box[3]), 1.0)
        return movement, movement / face_size

    def _append_distinct_face(
        self, face_boxes, face_confidences, new_box, new_confidence
    ):
        """Ignore a duplicate detection that strongly overlaps an existing face."""
        for index, existing_box in enumerate(face_boxes):
            if self._box_iou(existing_box, new_box) >= 0.60:
                if new_confidence > face_confidences[index]:
                    face_boxes[index] = new_box
                    face_confidences[index] = new_confidence
                return
        face_boxes.append(new_box)
        face_confidences.append(new_confidence)

    def _box_iou(self, first_box, second_box):
        """Return the shared area divided by the total area of two boxes."""
        first_x, first_y, first_width, first_height = first_box
        second_x, second_y, second_width, second_height = second_box
        left = max(first_x, second_x)
        top = max(first_y, second_y)
        right = min(first_x + first_width, second_x + second_width)
        bottom = min(first_y + first_height, second_y + second_height)
        if right <= left or bottom <= top:
            return 0.0

        intersection_area = (right - left) * (bottom - top)
        total_area = (
            first_width * first_height
            + second_width * second_height
            - intersection_area
        )
        return intersection_area / max(total_area, 1)

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
            color = (0, 0, 255) if len(boxes) > 1 or is_outside else (0, 255, 0)
            label = f"Face {index + 1}: {confidence:.0%}"
            label_position = (x, max(22, y - 8))
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                color,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (label_position[0] + 2, label_position[1] + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (20, 20, 20),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )

    def release(self):
        """Release the native MediaPipe Tasks resources."""
        self.face_detector.close()
        self.face_landmarker.close()
