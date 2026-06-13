"""Tests for report data loading, summaries, and CSV export."""

from datetime import date
import tempfile
import unittest
from pathlib import Path

from monitoring.alert_manager import AlertManager
from reports.report_generator import ReportGenerator


class ReportGeneratorTests(unittest.TestCase):
    """Verify daily reports count warnings and critical alerts correctly."""

    def test_daily_summary_and_csv_export(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            log_path = temporary_path / "activity_log.csv"
            report_directory = temporary_path / "reports"
            manager = AlertManager(
                log_file=log_path,
                cooldown_seconds=0,
                logging_enabled=True,
            )
            manager.info("MONITORING_STARTED", "Session started.")
            manager.warning("LOOKING_AWAY", "Student looked away.")
            manager.critical("FACE_MISSING", "No face visible.")

            generator = ReportGenerator(log_path, report_directory)
            rows = generator.load_logs(date.today())
            summary = generator.build_summary(rows)
            csv_report_path = generator.export_csv(date.today())
            pdf_report_path = generator.export_pdf(date.today())

            self.assertEqual(summary["total_alerts"], 2)
            self.assertEqual(summary["warning_count"], 1)
            self.assertEqual(summary["critical_count"], 1)
            self.assertEqual(summary["face_violations"], 2)
            self.assertTrue(csv_report_path.exists())
            self.assertTrue(pdf_report_path.exists())
            self.assertGreater(pdf_report_path.stat().st_size, 1000)
            self.assertIn(
                "Daily Report",
                csv_report_path.read_text(encoding="utf-8-sig"),
            )


if __name__ == "__main__":
    unittest.main()
