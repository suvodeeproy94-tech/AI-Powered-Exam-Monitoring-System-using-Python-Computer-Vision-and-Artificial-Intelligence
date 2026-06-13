"""Measure camera-frame quality and improve dark frames before detection."""

import cv2
import numpy as np


def prepare_detection_frame(frame, settings):
    """Return a detection frame and simple lighting and sharpness measurements."""
    if frame is None or frame.size == 0:
        raise ValueError("Frame quality check received an empty frame.")

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray_frame))
    blur_score = float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())
    is_low_light = brightness < settings.low_light_threshold
    is_blurry = blur_score < settings.blur_threshold

    prepared_frame = frame
    if settings.enhance_low_light and is_low_light:
        prepared_frame = _enhance_low_light_frame(frame)

    quality_results = {
        "frame_brightness": round(brightness, 2),
        "frame_blur_score": round(blur_score, 2),
        "is_low_light": is_low_light,
        "is_blurry": is_blurry,
        "frame_was_enhanced": prepared_frame is not frame,
    }
    return prepared_frame, quality_results


def _enhance_low_light_frame(frame):
    """Improve local contrast without changing the frame size or color format."""
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, color_a, color_b = cv2.split(lab_frame)
    contrast_enhancer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = contrast_enhancer.apply(lightness)
    enhanced_lab_frame = cv2.merge((enhanced_lightness, color_a, color_b))
    return cv2.cvtColor(enhanced_lab_frame, cv2.COLOR_LAB2BGR)
