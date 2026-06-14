"""Collect labeled MediaPipe hand landmarks for gesture-model training.

Example:
    python -m tools.collect_gesture_data --label "Open Palm" --participant P01
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2

from config import AppSettings, GESTURE_DATA_DIR, GESTURE_LABELS
from detection.hand_detector import HandDetector
from recognition.gesture_features import extract_gesture_features


def parse_arguments():
    """Read beginner-friendly collection options from the command line."""
    parser = argparse.ArgumentParser(description="Collect gesture landmark samples")
    parser.add_argument("--label", required=True, choices=GESTURE_LABELS)
    parser.add_argument("--participant", default="unknown")
    parser.add_argument("--lighting", default="normal")
    parser.add_argument("--angle", default="front")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=GESTURE_DATA_DIR)
    return parser.parse_args()


def main():
    """Open the webcam and save stable single-hand landmark samples."""
    arguments = parse_arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"{datetime.now():%Y%m%d_%H%M%S}_"
        f"{arguments.participant}_{arguments.label.replace(' ', '_')}.csv"
    )
    output_path = arguments.output / file_name
    settings = AppSettings(camera_index=arguments.camera)
    detector = HandDetector(settings)
    camera = cv2.VideoCapture(arguments.camera)
    saved_samples = 0
    frame_number = 0

    try:
        if not camera.isOpened():
            raise RuntimeError("The selected camera could not be opened.")

        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            field_names = [
                "label",
                "participant",
                "lighting",
                "angle",
            ] + [f"feature_{index}" for index in range(42)]
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()

            while saved_samples < arguments.samples:
                frame_ok, frame = camera.read()
                if not frame_ok or frame is None:
                    continue
                frame_number += 1
                frame = cv2.flip(frame, 1)
                annotated_frame, results = detector.process_frame(frame)

                # Saving every second frame reduces nearly identical samples.
                if results["hand_count"] == 1 and frame_number % 2 == 0:
                    landmarks = detector.get_landmark_list(0, frame.shape)
                    features = extract_gesture_features(landmarks)
                    row = {
                        "label": arguments.label,
                        "participant": arguments.participant,
                        "lighting": arguments.lighting,
                        "angle": arguments.angle,
                    }
                    row.update(
                        {
                            f"feature_{index}": feature_value
                            for index, feature_value in enumerate(features)
                        }
                    )
                    writer.writerow(row)
                    saved_samples += 1

                message = (
                    f"Show: {arguments.label} | "
                    f"Samples: {saved_samples}/{arguments.samples} | Q: stop"
                )
                cv2.putText(
                    annotated_frame,
                    message,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Gesture Data Collector", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        detector.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved_samples} samples to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
