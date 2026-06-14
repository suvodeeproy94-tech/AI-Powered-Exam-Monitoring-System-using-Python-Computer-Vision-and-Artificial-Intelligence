"""Evaluate the saved gesture model against collected labeled landmark data."""

import argparse
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from config import GESTURE_CLASSIFIER_MODEL, GESTURE_DATA_DIR
from recognition.trained_gesture_model import TrainedGestureModel
from tools.train_gesture_model import load_dataset


def main():
    """Print accuracy details without changing the saved model."""
    parser = argparse.ArgumentParser(description="Evaluate a trained gesture model")
    parser.add_argument("--data", type=Path, default=GESTURE_DATA_DIR)
    parser.add_argument("--model", type=Path, default=GESTURE_CLASSIFIER_MODEL)
    arguments = parser.parse_args()

    features, labels, _ = load_dataset(arguments.data)
    model_wrapper = TrainedGestureModel(arguments.model)
    if not model_wrapper.available:
        raise RuntimeError(f"Model could not be loaded: {model_wrapper.load_error}")
    if len(features) == 0:
        raise RuntimeError("No valid gesture samples were found.")

    predictions = model_wrapper.model.predict(features)
    print(classification_report(labels, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(labels, predictions, labels=model_wrapper.model.classes_))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
