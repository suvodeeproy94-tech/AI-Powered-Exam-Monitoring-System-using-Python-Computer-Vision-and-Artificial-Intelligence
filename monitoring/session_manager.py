"""Store exam sessions and alert records in a local SQLite database.

This module keeps database work separate from the dashboard. The dashboard only
asks for a session to be created or finished, and this file handles the tables.
"""

from dataclasses import dataclass
from datetime import datetime
from contextlib import closing
from pathlib import Path
import sqlite3
from uuid import uuid4

from config import DATABASE_FILE


@dataclass(frozen=True)
class ExamSession:
    """Hold the student and exam details for one monitoring session."""

    session_id: str
    student_name: str
    roll_number: str
    exam_name: str
    subject_name: str
    started_at: datetime


class SessionManager:
    """Create, update, and store local exam monitoring sessions."""

    def __init__(self, database_file=DATABASE_FILE):
        self.database_file = Path(database_file)
        self.database_file.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _connect(self):
        """Open one short-lived SQLite connection."""
        return sqlite3.connect(self.database_file)

    def _create_tables(self):
        """Create database tables when the app runs for the first time."""
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS exam_sessions (
                    session_id TEXT PRIMARY KEY,
                    student_name TEXT NOT NULL,
                    roll_number TEXT NOT NULL,
                    exam_name TEXT NOT NULL,
                    subject_name TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_seconds REAL DEFAULT 0,
                    attention_percentage REAL DEFAULT 100,
                    maximum_risk_score REAL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    duration_seconds REAL DEFAULT 0,
                    risk_score REAL DEFAULT 0,
                    attention_percentage REAL DEFAULT 100,
                    evidence_path TEXT DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES exam_sessions(session_id)
                )
                """
            )
            connection.commit()

    def create_session(self, student_name, roll_number, exam_name, subject_name):
        """Save and return a new exam session before monitoring starts."""
        started_at = datetime.now()
        session = ExamSession(
            session_id=uuid4().hex,
            student_name=student_name.strip(),
            roll_number=roll_number.strip(),
            exam_name=exam_name.strip(),
            subject_name=subject_name.strip(),
            started_at=started_at,
        )
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO exam_sessions (
                    session_id,
                    student_name,
                    roll_number,
                    exam_name,
                    subject_name,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.student_name,
                    session.roll_number,
                    session.exam_name,
                    session.subject_name,
                    session.started_at.isoformat(timespec="seconds"),
                ),
            )
            connection.commit()
        return session

    def finish_session(self, session_id, session_stats):
        """Store final session statistics after monitoring stops."""
        if not session_id:
            return

        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE exam_sessions
                SET ended_at = ?,
                    duration_seconds = ?,
                    attention_percentage = ?,
                    maximum_risk_score = ?
                WHERE session_id = ?
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    float(session_stats.get("session_duration_seconds", 0.0)),
                    float(session_stats.get("attention_percentage", 100.0)),
                    float(session_stats.get("maximum_risk_score", 0.0)),
                    session_id,
                ),
            )
            connection.commit()

    def store_alert(self, alert):
        """Store one alert row in SQLite for future review screens."""
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO alert_records (
                    session_id,
                    date,
                    time,
                    event_type,
                    alert_type,
                    description,
                    duration_seconds,
                    risk_score,
                    attention_percentage,
                    evidence_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.session_id,
                    alert.timestamp.strftime("%Y-%m-%d"),
                    alert.timestamp.strftime("%H:%M:%S"),
                    alert.event_type,
                    alert.level,
                    alert.description,
                    float(alert.duration_seconds),
                    float(alert.risk_score),
                    float(alert.attention_percentage),
                    alert.evidence_path,
                ),
            )
            connection.commit()

    def get_session(self, session_id):
        """Return one session row as a dictionary for reports or review."""
        if not session_id:
            return None

        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM exam_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None
