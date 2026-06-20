"""Generate daily CSV and PDF monitoring reports from the activity log.

The report generator reads the same CSV file written by AlertManager. It does
not depend on the live dashboard, so reports can also be created after an exam.
"""

import csv
from datetime import date, datetime
from html import escape
from pathlib import Path

from config import ACTIVITY_LOG_FILE, REPORT_DIR

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


FACE_EVENTS = {
    "FACE_MISSING",
    "MULTIPLE_FACES",
    "FACE_OUTSIDE_FRAME",
    "LOOKING_AWAY",
    "EYES_CLOSED",
    "HAND_COVERING_FACE",
}
GESTURE_EVENTS = {"SUSPICIOUS_GESTURE"}
MOVEMENT_EVENTS = {"EXCESSIVE_HAND_MOVEMENT", "FREQUENT_MOVEMENT"}
GADGET_EVENTS = {"DIGITAL_GADGET_DETECTED"}


class ReportGenerator:
    """Read activity records and export one selected day."""

    def __init__(self, log_file=ACTIVITY_LOG_FILE, report_dir=REPORT_DIR):
        self.log_file = Path(log_file)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def load_logs(self, target_date=None):
        """Load valid activity rows and optionally filter them by date."""
        if not self.log_file.exists() or self.log_file.stat().st_size == 0:
            return []

        activity_rows = []
        with self.log_file.open("r", newline="", encoding="utf-8-sig") as csv_file:
            for original_row in csv.DictReader(csv_file):
                normalized_row = self._normalize_log_row(original_row)
                if normalized_row is None:
                    continue
                if target_date and normalized_row["date"] != target_date.isoformat():
                    continue
                activity_rows.append(normalized_row)

        return activity_rows

    def _normalize_log_row(self, row):
        """Support both the current log columns and the old draft format."""
        if row.get("date") and row.get("time"):
            row_date = row["date"].strip()
            row_time = row["time"].strip()
            alert_type = row.get("alert_type", "").strip()
        elif row.get("timestamp"):
            try:
                timestamp = datetime.strptime(
                    row["timestamp"].strip(), "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                return None
            row_date = timestamp.strftime("%Y-%m-%d")
            row_time = timestamp.strftime("%H:%M:%S")
            alert_type = row.get("level", "").strip()
        else:
            return None

        try:
            datetime.strptime(f"{row_date} {row_time}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

        return {
            "session_id": row.get("session_id", "").strip(),
            "student_name": row.get("student_name", "").strip(),
            "roll_number": row.get("roll_number", "").strip(),
            "exam_name": row.get("exam_name", "").strip(),
            "subject_name": row.get("subject_name", "").strip(),
            "date": row_date,
            "time": row_time,
            "event_type": row.get("event_type", "UNKNOWN").strip() or "UNKNOWN",
            "alert_type": alert_type or "INFO",
            "description": row.get("description", "").strip(),
            "duration_seconds": self._safe_float(row.get("duration_seconds")),
            "risk_score": self._safe_float(row.get("risk_score")),
            "attention_percentage": self._safe_float(
                row.get("attention_percentage"), 100.0
            ),
            "evidence_path": row.get("evidence_path", "").strip(),
        }

    def _safe_float(self, value, default=0.0):
        """Convert optional old or new CSV values without breaking reports."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def build_summary(self, activity_rows):
        """Calculate the counts required by the daily monitoring report."""
        warning_count = self._count_by_value(activity_rows, "alert_type", "WARNING")
        critical_count = self._count_by_value(activity_rows, "alert_type", "CRITICAL")
        information_count = self._count_by_value(activity_rows, "alert_type", "INFO")

        summary_rows = [
            row for row in activity_rows if row["event_type"] == "SESSION_SUMMARY"
        ]
        session_duration = sum(
            row["duration_seconds"] for row in summary_rows
        )
        if session_duration > 0:
            attention_percentage = sum(
                row["attention_percentage"] * row["duration_seconds"]
                for row in summary_rows
            ) / session_duration
        elif summary_rows:
            attention_percentage = sum(
                row["attention_percentage"] for row in summary_rows
            ) / len(summary_rows)
        else:
            attention_percentage = 100.0
        maximum_risk_score = max(
            (row["risk_score"] for row in activity_rows), default=0.0
        )
        session_count = len(
            {
                row["session_id"]
                for row in activity_rows
                if row.get("session_id")
            }
        )

        return {
            "exam_sessions": session_count,
            "total_alerts": warning_count + critical_count,
            "information_events": information_count,
            "warning_count": warning_count,
            "critical_count": critical_count,
            "face_violations": self._count_event_group(activity_rows, FACE_EVENTS),
            "gesture_violations": self._count_event_group(activity_rows, GESTURE_EVENTS),
            "gadget_violations": self._count_event_group(activity_rows, GADGET_EVENTS),
            "movement_violations": self._count_event_group(activity_rows, MOVEMENT_EVENTS),
            "evidence_snapshots": sum(
                1 for row in activity_rows if row["evidence_path"]
            ),
            "session_duration_seconds": round(session_duration, 1),
            "attention_percentage": round(attention_percentage, 1),
            "maximum_risk_score": round(maximum_risk_score, 1),
            "face_missing_duration_seconds": round(
                self._sum_event_duration(activity_rows, "FACE_MISSING"), 1
            ),
            "look_away_duration_seconds": round(
                self._sum_event_duration(activity_rows, "LOOKING_AWAY"), 1
            ),
            "eyes_closed_duration_seconds": round(
                self._sum_event_duration(activity_rows, "EYES_CLOSED"), 1
            ),
        }

    def _count_by_value(self, rows, key, expected_value):
        """Count rows where one column equals the expected value."""
        return sum(1 for row in rows if row.get(key) == expected_value)

    def _count_event_group(self, rows, event_names):
        """Count rows whose event type belongs to one report category."""
        return sum(1 for row in rows if row.get("event_type") in event_names)

    def _sum_event_duration(self, rows, event_name):
        """Add confirmed durations for one event across the selected day."""
        return sum(
            row.get("duration_seconds", 0.0)
            for row in rows
            if row.get("event_type") == event_name
        )

    def export_csv(self, target_date=None):
        """Create a readable daily CSV report and return its full path."""
        report_date = target_date or date.today()
        activity_rows = self.load_logs(report_date)
        summary = self.build_summary(activity_rows)
        report_path = self.report_dir / f"exam_report_{report_date:%Y-%m-%d}.csv"

        with report_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["AI Exam Monitoring System - Daily Report"])
            writer.writerow(["Report Date", report_date.isoformat()])
            writer.writerow(["Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])
            writer.writerow(["Summary Metric", "Count"])
            for metric_name, metric_value in summary.items():
                writer.writerow([metric_name.replace("_", " ").title(), metric_value])

            writer.writerow([])
            writer.writerow(
                [
                    "Date",
                    "Time",
                    "Student Name",
                    "Roll Number",
                    "Exam Name",
                    "Subject Name",
                    "Alert Type",
                    "Event Type",
                    "Duration Seconds",
                    "Risk Score",
                    "Attention Percentage",
                    "Evidence Path",
                    "Description",
                ]
            )
            for row in activity_rows:
                writer.writerow(
                    [
                        row["date"],
                        row["time"],
                        row["student_name"],
                        row["roll_number"],
                        row["exam_name"],
                        row["subject_name"],
                        row["alert_type"],
                        row["event_type"],
                        row["duration_seconds"],
                        row["risk_score"],
                        row["attention_percentage"],
                        row["evidence_path"],
                        row["description"],
                    ]
                )

        return report_path.resolve()

    def export_pdf(self, target_date=None):
        """Create a formatted daily PDF report and return its full path."""
        if not PDF_AVAILABLE:
            raise RuntimeError(
                "PDF export requires reportlab. Run: "
                "python -m pip install reportlab"
            )

        report_date = target_date or date.today()
        activity_rows = self.load_logs(report_date)
        summary = self.build_summary(activity_rows)
        report_path = self.report_dir / f"exam_report_{report_date:%Y-%m-%d}.pdf"

        document = SimpleDocTemplate(
            str(report_path),
            pagesize=A4,
            leftMargin=1.5 * cm,
            rightMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
            title="AI Exam Monitoring System - Daily Report",
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ExamReportTitle",
            parent=styles["Title"],
            fontSize=18,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=8,
        )
        small_style = ParagraphStyle(
            "ExamReportSmall",
            parent=styles["BodyText"],
            fontSize=7,
            leading=9,
        )

        story = [
            Paragraph("AI Exam Monitoring System", title_style),
            Paragraph(
                f"Daily Monitoring Report - {report_date.strftime('%d %B %Y')}",
                styles["Heading2"],
            ),
            Paragraph(
                f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["BodyText"],
            ),
            Paragraph(
                self._build_session_line(activity_rows),
                styles["BodyText"],
            ),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94a3b8")),
            Spacer(1, 0.35 * cm),
            Paragraph("Monitoring Summary", styles["Heading2"]),
        ]

        summary_table_data = [["Metric", "Count"]]
        summary_table_data.extend(
            [metric.replace("_", " ").title(), str(value)]
            for metric, value in summary.items()
        )
        summary_table = Table(
            summary_table_data,
            colWidths=[13 * cm, 4 * cm],
            repeatRows=1,
        )
        self._style_table(summary_table, font_size=9)
        story.extend([summary_table, Spacer(1, 0.5 * cm)])

        story.append(Paragraph("Activity Details", styles["Heading2"]))
        if activity_rows:
            activity_table_data = [
                ["Time", "Student", "Type", "Event", "Risk", "Evidence", "Description"]
            ]
            for row in activity_rows:
                activity_table_data.append(
                    [
                        row["time"],
                        Paragraph(
                            escape(self._student_label(row)),
                            small_style,
                        ),
                        row["alert_type"],
                        row["event_type"].replace("_", " ").title(),
                        f"{row['risk_score']:.0f}",
                        "Yes" if row["evidence_path"] else "No",
                        Paragraph(escape(row["description"]), small_style),
                    ]
                )

            activity_table = Table(
                activity_table_data,
                colWidths=[
                    1.5 * cm,
                    2.5 * cm,
                    1.6 * cm,
                    3.2 * cm,
                    1.1 * cm,
                    1.2 * cm,
                    5.9 * cm,
                ],
                repeatRows=1,
            )
            self._style_table(activity_table, font_size=7)
            story.append(activity_table)
        else:
            story.append(
                Paragraph("No monitoring activity was recorded for this date.", styles["BodyText"])
            )

        document.build(story)
        return report_path.resolve()

    def _build_session_line(self, activity_rows):
        """Create a short student and exam line for the PDF header."""
        sessions = {
            row["session_id"]: row
            for row in activity_rows
            if row.get("session_id")
        }
        if not sessions:
            return "Session: No session details were recorded for this report."
        if len(sessions) > 1:
            return f"Sessions covered: {len(sessions)} exam sessions."

        session_row = next(iter(sessions.values()))
        return (
            "Session: "
            f"{escape(session_row['student_name'] or 'Unknown Student')} "
            f"({escape(session_row['roll_number'] or 'No Roll Number')}) | "
            f"{escape(session_row['exam_name'] or 'Unknown Exam')} | "
            f"{escape(session_row['subject_name'] or 'Unknown Subject')}"
        )

    def _student_label(self, row):
        """Return a compact student label for report tables."""
        student_name = row.get("student_name") or "Unknown"
        roll_number = row.get("roll_number") or "No Roll"
        return f"{student_name}\n{roll_number}"

    def _style_table(self, table, font_size):
        """Apply the same professional style to summary and activity tables."""
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), font_size),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
