"""Tests for local exam session database storage."""

import tempfile
import unittest
from pathlib import Path

from monitoring.alert_manager import AlertManager
from monitoring.session_manager import SessionManager


class SessionManagerTests(unittest.TestCase):
    """Verify exam sessions and related alerts are stored in SQLite."""

    def test_session_is_created_and_finished(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "exam_monitoring.db"
            session_manager = SessionManager(database_path)
            session = session_manager.create_session(
                "Student One",
                "ROLL-100",
                "Final Exam",
                "Computer Vision",
            )

            session_manager.finish_session(
                session.session_id,
                {
                    "session_duration_seconds": 90,
                    "attention_percentage": 92.5,
                    "maximum_risk_score": 35,
                },
            )

            saved_session = session_manager.get_session(session.session_id)

            self.assertEqual(saved_session["student_name"], "Student One")
            self.assertEqual(saved_session["roll_number"], "ROLL-100")
            self.assertEqual(saved_session["exam_name"], "Final Exam")
            self.assertEqual(saved_session["subject_name"], "Computer Vision")
            self.assertEqual(saved_session["duration_seconds"], 90)
            self.assertEqual(saved_session["attention_percentage"], 92.5)
            self.assertEqual(saved_session["maximum_risk_score"], 35)

    def test_alert_is_stored_for_current_session(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            database_path = temporary_path / "exam_monitoring.db"
            log_path = temporary_path / "activity_log.csv"
            session_manager = SessionManager(database_path)
            session = session_manager.create_session(
                "Student Two",
                "ROLL-200",
                "Unit Test",
                "Python",
            )
            alert_manager = AlertManager(log_path, cooldown_seconds=0)
            alert_manager.set_session_manager(session_manager)
            alert_manager.set_current_session(session)

            alert = alert_manager.critical("FACE_MISSING", "No face visible.")

            self.assertEqual(alert.session_id, session.session_id)
            self.assertEqual(alert.student_name, "Student Two")


if __name__ == "__main__":
    unittest.main()
