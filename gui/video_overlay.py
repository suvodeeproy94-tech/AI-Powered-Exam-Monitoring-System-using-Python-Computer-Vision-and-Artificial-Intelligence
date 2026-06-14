"""Draw clear face and gesture information on the live camera frame.

The dashboard sends face and gesture results to this file. Keeping the drawing
logic separate makes the text style easy to test and change later.
"""

import cv2


FACE_OK_COLOR = (0, 255, 0)
FACE_ALERT_COLOR = (0, 0, 255)
GESTURE_COLOR = (0, 255, 255)
TEXT_SHADOW_COLOR = (20, 20, 20)


def build_face_summary(face_count):
    """Return simple face-count text and its display color."""
    if face_count == 0:
        return "No face detected", FACE_ALERT_COLOR
    if face_count == 1:
        return "One face detected", FACE_OK_COLOR
    return f"{face_count} faces detected", FACE_ALERT_COLOR


def build_gesture_summary(gesture_results):
    """Return one readable line for all currently detected hand gestures."""
    if not gesture_results:
        return "Hand gesture: none"

    gesture_parts = []
    for result in gesture_results:
        hand_name = result.get("hand", "Hand")
        gesture_name = result.get("gesture", "Unknown Gesture")
        confidence = float(result.get("confidence", 0.0))

        if gesture_name == "Unknown Gesture":
            readable_gesture = "detecting..."
        else:
            readable_gesture = f"{gesture_name} {confidence:.0%}"

        if len(gesture_results) == 1:
            gesture_parts.append(readable_gesture)
        else:
            gesture_parts.append(f"{hand_name}: {readable_gesture}")

    heading = "Hand gesture" if len(gesture_results) == 1 else "Hand gestures"
    return f"{heading}: {' | '.join(gesture_parts)}"


def draw_monitoring_overlay(frame, face_count, gesture_results):
    """Draw the reference-style face and gesture summary on one frame."""
    if frame is None or frame.size == 0:
        raise ValueError("Cannot draw monitoring information on an empty frame.")

    frame_width = frame.shape[1]
    font_scale = max(0.62, min(0.90, frame_width / 850))
    text_thickness = 2
    line_height = max(28, int(38 * font_scale))
    start_x = 14
    start_y = max(28, int(36 * font_scale))

    face_text, face_color = build_face_summary(face_count)
    gesture_text = build_gesture_summary(gesture_results)
    _draw_text_with_shadow(
        frame,
        face_text,
        (start_x, start_y),
        face_color,
        font_scale,
        text_thickness,
    )
    _draw_text_with_shadow(
        frame,
        gesture_text,
        (start_x, start_y + line_height),
        GESTURE_COLOR,
        font_scale,
        text_thickness,
    )
    return frame


def _draw_text_with_shadow(frame, text, position, color, scale, thickness):
    """Draw dark text behind the colored text so it stays readable."""
    shadow_position = (position[0] + 2, position[1] + 2)
    cv2.putText(
        frame,
        text,
        shadow_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        TEXT_SHADOW_COLOR,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
