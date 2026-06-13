"""Start the AI Exam Monitoring System desktop application.

This file checks the local Python environment before importing the GUI. That
gives beginners a clear setup message instead of a long import traceback.
"""

from importlib.util import find_spec
import sys

from config import FACE_DETECTOR_MODEL, FACE_LANDMARKER_MODEL, HAND_LANDMARKER_MODEL


REQUIRED_MODULES = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "PIL": "Pillow",
    "customtkinter": "customtkinter",
    "reportlab": "reportlab",
}

REQUIRED_MODEL_FILES = (
    FACE_DETECTOR_MODEL,
    FACE_LANDMARKER_MODEL,
    HAND_LANDMARKER_MODEL,
)


def get_runtime_problems():
    """Return clear environment problems that prevent the GUI from starting."""
    problems = []

    if sys.version_info < (3, 9):
        problems.append("Python 3.9 or newer is required.")

    missing_packages = [
        package_name
        for module_name, package_name in REQUIRED_MODULES.items()
        if find_spec(module_name) is None
    ]
    if missing_packages:
        problems.append(
            "Missing packages: " + ", ".join(sorted(missing_packages)) + ". "
            "Run: python -m pip install -r requirements.txt"
        )

    missing_models = [str(path) for path in REQUIRED_MODEL_FILES if not path.exists()]
    if missing_models:
        problems.append("Missing MediaPipe model files: " + ", ".join(missing_models))

    return problems


def main():
    """Validate dependencies and launch the CustomTkinter dashboard."""
    runtime_problems = get_runtime_problems()
    if runtime_problems:
        print("AI Exam Monitoring System could not start:\n")
        for problem in runtime_problems:
            print(f"- {problem}")
        return 1

    from gui.dashboard import Dashboard

    application = Dashboard()
    application.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
