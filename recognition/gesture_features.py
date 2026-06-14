"""Convert 21 hand landmarks into scale- and rotation-stable model features."""

import math


def extract_gesture_features(landmarks):
    """Return 42 normalized x/y values for training or model prediction."""
    if len(landmarks) != 21:
        raise ValueError("Exactly 21 hand landmarks are required.")

    points = [(float(point[0]), float(point[1])) for point in landmarks]
    wrist_x, wrist_y = points[0]
    centered_points = [
        (point_x - wrist_x, point_y - wrist_y)
        for point_x, point_y in points
    ]

    middle_mcp_x, middle_mcp_y = centered_points[9]
    rotation_angle = math.atan2(middle_mcp_y, middle_mcp_x) - math.pi / 2
    cosine_value = math.cos(-rotation_angle)
    sine_value = math.sin(-rotation_angle)
    rotated_points = [
        (
            point_x * cosine_value - point_y * sine_value,
            point_x * sine_value + point_y * cosine_value,
        )
        for point_x, point_y in centered_points
    ]

    scale = max(
        math.hypot(point_x, point_y)
        for point_x, point_y in rotated_points
    )
    scale = max(scale, 0.0001)
    features = []
    for point_x, point_y in rotated_points:
        features.extend((point_x / scale, point_y / scale))
    return features
