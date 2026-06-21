"""Detect phone-like digital gadgets with simple OpenCV shape rules.

This detector is intentionally lightweight. It does not claim to identify every
mobile phone perfectly. It searches for rectangular objects that look like a
phone or small tablet and asks the behavior monitor to confirm them over time.
"""

import cv2
import numpy as np


GADGET_BOX_COLOR = (0, 165, 255)


class GadgetDetector:
    """Find possible mobile phones or digital gadgets in one camera frame."""

    def __init__(self, settings):
        self.settings = settings

    def process_frame(self, frame, drawing_frame=None, face_boxes=None, hand_boxes=None):
        """Return an annotated frame and possible gadget detections."""
        if frame is None or frame.size == 0:
            raise ValueError("GadgetDetector received an empty camera frame.")

        annotated_frame = (
            drawing_frame.copy() if drawing_frame is not None else frame.copy()
        )
        if not self.settings.digital_gadget_detection_enabled:
            return annotated_frame, self._empty_results()

        frame_height, frame_width = frame.shape[:2]
        candidates = self._find_candidates(frame, frame_width, frame_height)
        filtered_candidates = self._remove_body_overlaps(
            candidates,
            face_boxes or [],
            hand_boxes or [],
        )

        for candidate in filtered_candidates:
            self._draw_candidate(annotated_frame, candidate)

        return annotated_frame, {
            "gadget_count": len(filtered_candidates),
            "gadget_boxes": [candidate["box"] for candidate in filtered_candidates],
            "gadget_confidences": [
                candidate["confidence"] for candidate in filtered_candidates
            ],
            "gadget_detected": bool(filtered_candidates),
        }

    def _empty_results(self):
        """Return a complete result object when no gadget is detected."""
        return {
            "gadget_count": 0,
            "gadget_boxes": [],
            "gadget_confidences": [],
            "gadget_detected": False,
        }

    def _find_candidates(self, frame, frame_width, frame_height):
        """Find rectangular objects that match common phone proportions."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv2.Canny(blurred_frame, 45, 120)
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        frame_area = float(frame_width * frame_height)
        candidates = []
        for contour in contours:
            box_x, box_y, box_width, box_height = cv2.boundingRect(contour)
            if self._touches_frame_border(
                box_x,
                box_y,
                box_width,
                box_height,
                frame_width,
                frame_height,
            ):
                continue

            area_ratio = (box_width * box_height) / frame_area
            if not self._area_is_valid(area_ratio):
                continue

            shape_score, rectangular_fill = self._shape_score(
                contour,
                box_width,
                box_height,
            )
            if rectangular_fill < self.settings.digital_gadget_min_rectangular_fill:
                continue

            visual_score = self._visual_score(
                frame,
                gray_frame,
                box_x,
                box_y,
                box_width,
                box_height,
            )
            if visual_score == 0.0:
                continue

            confidence = round(0.60 * shape_score + 0.40 * visual_score, 2)
            if confidence < self.settings.digital_gadget_min_confidence:
                continue

            candidates.append(
                {
                    "box": (box_x, box_y, box_width, box_height),
                    "confidence": confidence,
                    "area_ratio": area_ratio,
                }
            )

        return self._deduplicate(candidates)

    def _area_is_valid(self, area_ratio):
        """Check whether the object size is practical for a visible gadget."""
        return (
            self.settings.digital_gadget_min_area_ratio
            <= area_ratio
            <= self.settings.digital_gadget_max_area_ratio
        )

    def _touches_frame_border(
        self,
        box_x,
        box_y,
        box_width,
        box_height,
        frame_width,
        frame_height,
    ):
        """Reject partial edge objects because they are often arms or furniture."""
        border_margin = 3
        return (
            box_x <= border_margin
            or box_y <= border_margin
            or box_x + box_width >= frame_width - border_margin
            or box_y + box_height >= frame_height - border_margin
        )

    def _shape_score(self, contour, box_width, box_height):
        """Score rectangular shape and phone/tablet aspect ratio."""
        if box_width <= 0 or box_height <= 0:
            return 0.0, 0.0

        aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))
        if aspect_ratio < 1.25 or aspect_ratio > 3.2:
            return 0.0, 0.0

        contour_area = max(cv2.contourArea(contour), 1.0)
        box_area = max(float(box_width * box_height), 1.0)
        rectangular_fill = min(contour_area / box_area, 1.0)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) > 6:
            return 0.0, rectangular_fill

        corner_score = 1.0 if len(approx) == 4 else 0.75
        aspect_score = 1.0 - min(abs(aspect_ratio - 1.8) / 1.8, 1.0)
        score = (
            0.45 * rectangular_fill
            + 0.35 * corner_score
            + 0.20 * aspect_score
        )
        return max(0.0, min(1.0, score)), rectangular_fill

    def _visual_score(self, frame, gray_frame, box_x, box_y, box_width, box_height):
        """Score gadget-like color while rejecting human skin regions."""
        color_crop = frame[box_y : box_y + box_height, box_x : box_x + box_width]
        crop = gray_frame[box_y : box_y + box_height, box_x : box_x + box_width]
        if crop.size == 0 or color_crop.size == 0:
            return 0.0

        skin_ratio = self._skin_ratio(color_crop)
        if skin_ratio > self.settings.digital_gadget_max_skin_ratio:
            return 0.0

        dark_ratio = float(np.mean(crop <= 95))
        if dark_ratio < self.settings.digital_gadget_min_dark_ratio:
            return 0.0

        mean_brightness = float(np.mean(crop))
        contrast = float(np.std(crop))
        dark_body_score = 1.0 if mean_brightness <= 125 else 0.65
        dark_area_score = min(dark_ratio / 0.45, 1.0)
        contrast_score = min(contrast / 55.0, 1.0)
        score = (
            0.35 * dark_body_score
            + 0.35 * dark_area_score
            + 0.30 * contrast_score
        )
        return max(0.0, min(1.0, score))

    def _skin_ratio(self, color_crop):
        """Return how much of one crop looks like skin color."""
        ycrcb_crop = cv2.cvtColor(color_crop, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb_crop, lower_skin, upper_skin)
        return float(np.mean(skin_mask > 0))

    def _remove_body_overlaps(self, candidates, face_boxes, hand_boxes):
        """Avoid marking the face or hand itself as a gadget."""
        filtered_candidates = []
        body_boxes = list(face_boxes) + list(hand_boxes)
        for candidate in candidates:
            candidate_box = candidate["box"]
            largest_overlap = max(
                (
                    self._overlap_ratio(candidate_box, body_box)
                    for body_box in body_boxes
                ),
                default=0.0,
            )
            if largest_overlap <= 0.55:
                filtered_candidates.append(candidate)
        return filtered_candidates

    def _deduplicate(self, candidates):
        """Keep the highest confidence candidate when boxes overlap."""
        sorted_candidates = sorted(
            candidates,
            key=lambda item: item["confidence"],
            reverse=True,
        )
        selected_candidates = []
        for candidate in sorted_candidates:
            overlaps_existing = any(
                self._overlap_ratio(candidate["box"], selected["box"]) > 0.40
                for selected in selected_candidates
            )
            if not overlaps_existing:
                selected_candidates.append(candidate)
        return selected_candidates[:2]

    def _overlap_ratio(self, first_box, second_box):
        """Return overlap area divided by the smaller box area."""
        first_x, first_y, first_width, first_height = first_box
        second_x, second_y, second_width, second_height = second_box
        left = max(first_x, second_x)
        top = max(first_y, second_y)
        right = min(first_x + first_width, second_x + second_width)
        bottom = min(first_y + first_height, second_y + second_height)
        if right <= left or bottom <= top:
            return 0.0

        overlap_area = float((right - left) * (bottom - top))
        first_area = max(float(first_width * first_height), 1.0)
        second_area = max(float(second_width * second_height), 1.0)
        return overlap_area / min(first_area, second_area)

    def _draw_candidate(self, frame, candidate):
        """Draw a clear orange box around a possible digital gadget."""
        box_x, box_y, box_width, box_height = candidate["box"]
        confidence = candidate["confidence"]
        cv2.rectangle(
            frame,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            GADGET_BOX_COLOR,
            2,
        )
        cv2.putText(
            frame,
            f"Digital gadget {confidence:.0%}",
            (box_x, max(18, box_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            GADGET_BOX_COLOR,
            2,
            cv2.LINE_AA,
        )
