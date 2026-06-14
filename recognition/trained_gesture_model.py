"""Load and use an optional trained gesture classifier.

The application continues using geometric rules when no trained model exists.
This allows the project to work immediately and improve after real gesture data
has been collected.
"""

from pathlib import Path

from config import GESTURE_CLASSIFIER_MODEL, GESTURE_LABELS
from recognition.gesture_features import extract_gesture_features

try:
    import joblib
except ImportError:
    joblib = None


class TrainedGestureModel:
    """Provide safe optional predictions from a scikit-learn model file."""

    def __init__(self, model_path=GESTURE_CLASSIFIER_MODEL):
        self.model_path = Path(model_path)
        self.model = None
        self.metadata = {}
        self.load_error = ""
        self.load()

    @property
    def available(self):
        """Return True only when a valid model was loaded."""
        return self.model is not None

    def load(self):
        """Load a saved model bundle without stopping the application on errors."""
        self.model = None
        self.metadata = {}
        self.load_error = ""
        if joblib is None:
            self.load_error = "joblib is unavailable"
            return False
        if not self.model_path.exists():
            self.load_error = "trained model file does not exist"
            return False

        try:
            saved_bundle = joblib.load(self.model_path)
            if isinstance(saved_bundle, dict):
                self.model = saved_bundle.get("model")
                self.metadata = saved_bundle.get("metadata", {})
            else:
                self.model = saved_bundle
            if self.model is None or not hasattr(self.model, "predict_proba"):
                raise ValueError("saved object is not a probability classifier")
        except Exception as error:
            # A damaged or incompatible local model must never stop monitoring.
            self.model = None
            self.load_error = str(error)
            return False
        return True

    def predict(self, landmarks):
        """Return gesture name and confidence, or Unknown when unavailable."""
        if not self.available:
            return "Unknown Gesture", 0.0

        try:
            features = extract_gesture_features(landmarks)
            probabilities = self.model.predict_proba([features])[0]
            best_index = int(probabilities.argmax())
            gesture_name = str(self.model.classes_[best_index])
            confidence = float(probabilities[best_index])
        except Exception:
            # Prediction falls back to geometric rules in the caller.
            return "Unknown Gesture", 0.0

        if gesture_name not in GESTURE_LABELS:
            return "Unknown Gesture", 0.0
        return gesture_name, confidence
