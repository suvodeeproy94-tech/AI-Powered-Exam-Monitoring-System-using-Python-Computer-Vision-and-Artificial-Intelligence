"""Tests for alert validation, cooldown, history, and CSV logging."""

import csv
import tempfile
import unittest
from pathlib import Path

from monitoring.alert_manager import AlertLevel, AlertManager, CSV_FIELDS


class AlertManagerTests(unittest.TestCase):
    """Verify alerts are stored once and use the documented CSV format."""

    def test_alert_is_logged_with_expected_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "activity_log.csv"
            manager = AlertManager(
                log_file=log_path,
                cooldown_seconds=0,
                logging_enabled=True,
            )

            manager.warning("LOOKING_AWAY", "Student looked away.")

            with log_path.open("r", newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, CSV_FIELDS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["alert_type"], AlertLevel.WARNING)
            self.assertEqual(rows[0]["event_type"], "LOOKING_AWAY")
            self.assertIn("risk_score", rows[0])
            self.assertIn("evidence_path", rows[0])

    def test_cooldown_blocks_immediate_duplicate(self):
        manager = AlertManager(cooldown_seconds=60, logging_enabled=False)

        first_alert = manager.critical("FACE_MISSING", "No face.")
        second_alert = manager.critical("FACE_MISSING", "No face.")

        self.assertIsNotNone(first_alert)
        self.assertIsNone(second_alert)
        self.assertEqual(manager.summary()[AlertLevel.CRITICAL], 1)

    def test_clear_history_keeps_manager_usable(self):
        manager = AlertManager(cooldown_seconds=0, logging_enabled=False)
        manager.info("STARTED", "Started.")
        manager.clear_history()
        manager.info("STARTED", "Started again.")

        self.assertEqual(len(manager.get_history()), 1)


if __name__ == "__main__":
    unittest.main()
