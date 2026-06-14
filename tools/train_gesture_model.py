"""Train and evaluate a Random Forest gesture classifier from collected CSVs."""

import argparse
import csv
import json
from pathlib import Path

from config import GESTURE_CLASSIFIER_MODEL, GESTURE_DATA_DIR, REPORT_DIR

import joblib
import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# Training runs as a command-line task, so charts must not require a GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_arguments():
    """Read training folders and output paths."""
    parser = argparse.ArgumentParser(description="Train the gesture classifier")
    parser.add_argument("--data", type=Path, default=GESTURE_DATA_DIR)
    parser.add_argument("--model", type=Path, default=GESTURE_CLASSIFIER_MODEL)
    parser.add_argument("--report", type=Path, default=REPORT_DIR / "gesture_model")
    return parser.parse_args()


def load_dataset(data_directory):
    """Load labeled feature rows from all collector CSV files."""
    feature_rows = []
    labels = []
    participants = []
    for csv_path in sorted(Path(data_directory).glob("*.csv")):
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                try:
                    features = [float(row[f"feature_{index}"]) for index in range(42)]
                except (KeyError, TypeError, ValueError):
                    continue
                feature_rows.append(features)
                labels.append(row.get("label", "Unknown Gesture"))
                participants.append(row.get("participant", "unknown"))
    return np.asarray(feature_rows), np.asarray(labels), np.asarray(participants)


def train_model(features, labels):
    """Train a readable and reliable Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features, labels)
    return model


def split_dataset(features, labels, participants):
    """Prefer testing on unseen people to avoid exaggerated accuracy values."""
    unique_participants = np.unique(participants)
    unique_labels = set(labels.tolist())
    if len(unique_participants) >= 3:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=42,
        )
        train_indexes, test_indexes = next(
            splitter.split(features, labels, groups=participants)
        )
        train_labels = labels[train_indexes]
        test_labels = labels[test_indexes]
        if set(train_labels.tolist()) == unique_labels and set(
            test_labels.tolist()
        ) == unique_labels:
            return (
                features[train_indexes],
                features[test_indexes],
                train_labels,
                test_labels,
                "participant_group_split",
            )

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    return (
        train_features,
        test_features,
        train_labels,
        test_labels,
        "stratified_sample_split",
    )


def save_confusion_matrix(matrix, class_names, output_path):
    """Save a visual confusion matrix for the project report."""
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set_xlabel("Predicted Gesture")
    axis.set_ylabel("Actual Gesture")
    axis.set_title("Gesture Recognition Confusion Matrix")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    """Validate data, train the model, and save transparent accuracy reports."""
    arguments = parse_arguments()
    features, labels, participants = load_dataset(arguments.data)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if len(features) < 40 or len(unique_labels) < 2:
        raise RuntimeError(
            "Collect at least 40 total samples from at least two gesture classes."
        )
    if min(label_counts) < 5:
        raise RuntimeError("Every gesture class needs at least five samples.")

    (
        train_features,
        test_features,
        train_labels,
        test_labels,
        split_strategy,
    ) = split_dataset(
        features, labels, participants
    )
    model = train_model(train_features, train_labels)
    predictions = model.predict(test_features)
    accuracy = float(accuracy_score(test_labels, predictions))
    report = classification_report(
        test_labels,
        predictions,
        labels=unique_labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_labels, predictions, labels=unique_labels)

    arguments.model.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.mkdir(parents=True, exist_ok=True)
    metadata = {
        "accuracy": accuracy,
        "sample_count": int(len(features)),
        "participants": sorted(set(participants.tolist())),
        "classes": unique_labels.tolist(),
        "test_strategy": split_strategy,
    }
    joblib.dump({"model": model, "metadata": metadata}, arguments.model)
    (arguments.report / "metrics.json").write_text(
        json.dumps({"metadata": metadata, "classification_report": report}, indent=2),
        encoding="utf-8",
    )
    np.savetxt(
        arguments.report / "confusion_matrix.csv",
        matrix,
        fmt="%d",
        delimiter=",",
    )
    save_confusion_matrix(
        matrix,
        unique_labels,
        arguments.report / "confusion_matrix.png",
    )
    print(f"Model accuracy: {accuracy:.2%}")
    print(f"Saved model: {arguments.model.resolve()}")
    print(f"Saved reports: {arguments.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
