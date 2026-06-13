"""Convert face, hand, and gesture measurements into exam alerts.

Each suspicious condition must remain true for a small number of frames before
an alert is created. This reduces false alerts caused by one blurred frame.
"""

from monitoring.alert_manager import AlertLevel


def boxes_overlap_ratio(face_box, hand_box):
    """Return how much of the face box is covered by the hand box."""
    face_x, face_y, face_width, face_height = face_box
    hand_x, hand_y, hand_width, hand_height = hand_box

    intersection_left = max(face_x, hand_x)
    intersection_top = max(face_y, hand_y)
    intersection_right = min(face_x + face_width, hand_x + hand_width)
    intersection_bottom = min(face_y + face_height, hand_y + hand_height)

    if intersection_right <= intersection_left or intersection_bottom <= intersection_top:
        return 0.0

    intersection_area = (
        intersection_right - intersection_left
    ) * (
        intersection_bottom - intersection_top
    )
    face_area = face_width * face_height
    if face_area <= 0:
        return 0.0
    return intersection_area / face_area


def count_landmarks_inside_box(landmark_points, box):
    """Count hand landmark points that are inside one face box."""
    box_x, box_y, box_width, box_height = box
    box_right = box_x + box_width
    box_bottom = box_y + box_height
    return sum(
        1
        for point_x, point_y in landmark_points
        if box_x <= point_x <= box_right and box_y <= point_y <= box_bottom
    )


class BehaviorMonitor:
    """Track suspicious conditions and send confirmed events to AlertManager."""

    def __init__(self, alert_manager, settings):
        self.alert_manager = alert_manager
        self.settings = settings
        self._condition_streaks = {}
        self._active_conditions = {}
        self._has_seen_face = False

        self.face_status = "Waiting for monitoring"
        self.hand_status = "No hands"
        self.gesture_status = "None"
        self.alert_status = "IDLE"
        self.alert_level = AlertLevel.INFO
        self.reset_session()

    def reset_session(self):
        """Clear temporary condition state and start fresh session statistics."""
        self._condition_streaks.clear()
        self._active_conditions.clear()
        self._has_seen_face = False
        self.stats = {
            "frames_processed": 0,
            "total_violations": 0,
            "face_missing_violations": 0,
            "multiple_face_violations": 0,
            "face_outside_violations": 0,
            "look_away_violations": 0,
            "hand_cover_violations": 0,
            "excessive_hand_violations": 0,
            "frequent_movement_violations": 0,
            "suspicious_gesture_violations": 0,
        }

    def analyse(self, face_results, hand_results, gesture_results):
        """Run every monitoring rule for one processed camera frame."""
        self.stats["frames_processed"] += 1
        fired_alerts = []

        fired_alerts.extend(self._check_face_count(face_results))
        fired_alerts.extend(self._check_face_outside_frame(face_results))
        fired_alerts.extend(self._check_look_away(face_results))
        fired_alerts.extend(self._check_hand_cover(face_results, hand_results))
        fired_alerts.extend(self._check_hand_movement(hand_results))
        fired_alerts.extend(self._check_frequent_movement(face_results))
        fired_alerts.extend(self._check_suspicious_gesture(gesture_results))

        self._update_display_status(face_results, hand_results, gesture_results)
        return fired_alerts

    def _condition_is_confirmed(self, condition_name, is_true, required_frames):
        """Count consecutive true frames for one monitoring condition."""
        if is_true:
            self._condition_streaks[condition_name] = (
                self._condition_streaks.get(condition_name, 0) + 1
            )
        else:
            self._condition_streaks[condition_name] = 0
            self._active_conditions[condition_name] = False

        confirmed = self._condition_streaks[condition_name] >= required_frames
        self._active_conditions[condition_name] = confirmed
        return confirmed

    def _record_violation(self, alert, statistic_name):
        """Increase session statistics only when a new alert was recorded."""
        if alert is None:
            return []
        self.stats[statistic_name] += 1
        self.stats["total_violations"] += 1
        return [alert]

    def _check_face_count(self, face_results):
        """Detect missing faces, multiple faces, and first successful detection."""
        face_count = face_results.get("face_count", 0)
        fired_alerts = []

        face_missing = self._condition_is_confirmed(
            "face_missing",
            face_count == 0,
            self.settings.face_missing_frames,
        )
        multiple_faces = self._condition_is_confirmed(
            "multiple_faces",
            face_count > self.settings.expected_face_count,
            self.settings.multiple_face_frames,
        )

        if face_missing:
            quality_hint = ""
            if face_results.get("is_low_light"):
                quality_hint = " Improve the room lighting or face the light source."
            elif face_results.get("is_blurry"):
                quality_hint = " Keep the camera and your face steady."
            alert = self.alert_manager.critical(
                "FACE_MISSING",
                "No face is visible. The student may have left the camera frame."
                + quality_hint,
            )
            fired_alerts.extend(
                self._record_violation(alert, "face_missing_violations")
            )
        elif multiple_faces:
            alert = self.alert_manager.critical(
                "MULTIPLE_FACES",
                f"{face_count} faces are visible. Only one student is expected.",
            )
            fired_alerts.extend(
                self._record_violation(alert, "multiple_face_violations")
            )
        elif face_count == self.settings.expected_face_count and not self._has_seen_face:
            self._has_seen_face = True
            alert = self.alert_manager.info(
                "FACE_DETECTED",
                "Face detected successfully.",
                force=True,
            )
            if alert:
                fired_alerts.append(alert)

        return fired_alerts

    def _check_face_outside_frame(self, face_results):
        """Alert when the main face remains too close to a frame edge."""
        is_outside = (
            face_results.get("face_count", 0) == 1
            and face_results.get("face_outside_frame", False)
        )
        confirmed = self._condition_is_confirmed(
            "face_outside",
            is_outside,
            self.settings.face_outside_frames,
        )
        if not confirmed:
            return []

        alert = self.alert_manager.warning(
            "FACE_OUTSIDE_FRAME",
            "The face is too close to the camera edge or partly outside the frame.",
        )
        return self._record_violation(alert, "face_outside_violations")

    def _check_look_away(self, face_results):
        """Alert when the student keeps looking away for several frames."""
        is_looking_away = (
            face_results.get("face_count", 0) == 1
            and face_results.get("is_looking_away", False)
        )
        confirmed = self._condition_is_confirmed(
            "looking_away",
            is_looking_away,
            self.settings.look_away_frames,
        )
        if not confirmed:
            return []

        alert = self.alert_manager.warning(
            "LOOKING_AWAY",
            "The student appears to be looking away from the exam screen.",
        )
        return self._record_violation(alert, "look_away_violations")

    def _check_hand_cover(self, face_results, hand_results):
        """Alert when a hand covers a meaningful part of a visible face."""
        face_boxes = face_results.get("face_bboxes", [])
        hand_boxes = hand_results.get("hand_bboxes", [])
        hand_landmark_groups = hand_results.get("hand_landmark_points", [])
        largest_overlap = 0.0
        largest_landmark_count = 0

        for face_box in face_boxes:
            for hand_index, hand_box in enumerate(hand_boxes):
                overlap = boxes_overlap_ratio(face_box, hand_box)
                landmark_points = (
                    hand_landmark_groups[hand_index]
                    if hand_index < len(hand_landmark_groups)
                    else []
                )
                landmark_count = count_landmarks_inside_box(
                    landmark_points, face_box
                )
                is_covering = (
                    overlap >= self.settings.face_cover_overlap_ratio
                    and landmark_count >= self.settings.hand_cover_min_landmarks
                )
                if is_covering and overlap > largest_overlap:
                    largest_overlap = overlap
                    largest_landmark_count = landmark_count

        confirmed = self._condition_is_confirmed(
            "hand_covering_face",
            largest_overlap > 0.0,
            self.settings.hand_cover_frames,
        )
        if not confirmed:
            return []

        alert = self.alert_manager.warning(
            "HAND_COVERING_FACE",
            f"A hand is covering about {largest_overlap:.0%} of the face area "
            f"with {largest_landmark_count} hand points over the face.",
        )
        return self._record_violation(alert, "hand_cover_violations")

    def _check_hand_movement(self, hand_results):
        """Alert when fast hand movement continues across several frames."""
        confirmed = self._condition_is_confirmed(
            "excessive_hand_movement",
            hand_results.get("excessive_movement", False),
            self.settings.excessive_hand_frames,
        )
        if not confirmed:
            return []

        alert = self.alert_manager.warning(
            "EXCESSIVE_HAND_MOVEMENT",
            "Fast hand movement continued for several camera frames.",
        )
        return self._record_violation(alert, "excessive_hand_violations")

    def _check_frequent_movement(self, face_results):
        """Alert when the student's face position changes rapidly and repeatedly."""
        movement_ratio = face_results.get("face_movement_ratio", 0.0)
        moving_frequently = (
            face_results.get("face_count", 0) == 1
            and movement_ratio >= self.settings.face_movement_ratio_threshold
        )
        confirmed = self._condition_is_confirmed(
            "frequent_movement",
            moving_frequently,
            self.settings.frequent_movement_frames,
        )
        if not confirmed:
            return []

        alert = self.alert_manager.warning(
            "FREQUENT_MOVEMENT",
            "Frequent body or head movement was detected.",
        )
        return self._record_violation(alert, "frequent_movement_violations")

    def _check_suspicious_gesture(self, gesture_results):
        """Alert when a suspicious gesture remains visible long enough to confirm."""
        suspicious_gestures = [
            result["gesture"]
            for result in gesture_results
            if result.get("is_suspicious")
        ]
        confirmed = self._condition_is_confirmed(
            "suspicious_gesture",
            bool(suspicious_gestures),
            self.settings.suspicious_gesture_frames,
        )
        if not confirmed:
            return []

        gesture_names = ", ".join(sorted(set(suspicious_gestures)))
        alert = self.alert_manager.warning(
            "SUSPICIOUS_GESTURE",
            f"Suspicious gesture detected: {gesture_names}.",
        )
        return self._record_violation(alert, "suspicious_gesture_violations")

    def _update_display_status(self, face_results, hand_results, gesture_results):
        """Prepare short status text values for the dashboard."""
        face_count = face_results.get("face_count", 0)
        if face_count == 0:
            if face_results.get("is_low_light"):
                self.face_status = "No face - low light"
            elif face_results.get("is_blurry"):
                self.face_status = "No face - blurry"
            else:
                self.face_status = "No face"
        elif face_count == 1 and face_results.get("face_outside_frame"):
            self.face_status = "Near frame edge"
        elif face_count == 1:
            if face_results.get("is_low_light"):
                self.face_status = "Face detected - low light"
            elif face_results.get("is_blurry"):
                self.face_status = "Face detected - blurry"
            else:
                self.face_status = "Face detected"
        else:
            self.face_status = f"{face_count} faces"

        hand_count = hand_results.get("hand_count", 0)
        if hand_count == 0:
            self.hand_status = "No hands"
        else:
            self.hand_status = f"{hand_count} hand" + ("s" if hand_count != 1 else "")

        if gesture_results:
            self.gesture_status = " | ".join(
                f"{result['gesture']} {result['confidence']:.0%}"
                for result in gesture_results
            )
        else:
            self.gesture_status = "None"

        critical_conditions = ("face_missing", "multiple_faces")
        warning_conditions = (
            "face_outside",
            "looking_away",
            "hand_covering_face",
            "excessive_hand_movement",
            "frequent_movement",
            "suspicious_gesture",
        )

        if any(self._active_conditions.get(name, False) for name in critical_conditions):
            self.alert_status = "CRITICAL"
            self.alert_level = AlertLevel.CRITICAL
        elif any(self._active_conditions.get(name, False) for name in warning_conditions):
            self.alert_status = "WARNING"
            self.alert_level = AlertLevel.WARNING
        else:
            self.alert_status = "NORMAL"
            self.alert_level = AlertLevel.INFO

    def get_snapshot(self):
        """Return simple status data that can safely move to the GUI thread."""
        return {
            "face_status": self.face_status,
            "hand_status": self.hand_status,
            "gesture_status": self.gesture_status,
            "alert_status": self.alert_status,
            "alert_level": self.alert_level,
            "stats": dict(self.stats),
        }
