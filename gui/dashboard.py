"""CustomTkinter dashboard for live AI exam monitoring.

The webcam and MediaPipe models run in a background thread. The Tkinter main
thread only displays the newest frame and status values, which keeps the user
interface responsive while detection is running.
"""

from datetime import datetime
import queue
import threading
import time
from tkinter import messagebox

import customtkinter as ctk
import cv2
from PIL import Image

from config import (
    COLOR_CRITICAL,
    COLOR_INFO,
    COLOR_NEUTRAL,
    COLOR_OK,
    COLOR_WARNING,
    FEED_HEIGHT,
    FEED_WIDTH,
    THEME,
    COLOR_SCHEME,
    WINDOW_TITLE,
    load_settings,
    save_settings,
)
from detection.face_detector import FaceDetector
from detection.frame_quality import prepare_detection_frame
from detection.hand_detector import HandDetector
from gui.components import AlertRow, StatusBadge
from gui.settings_dialog import SettingsDialog
from gui.video_overlay import draw_monitoring_overlay
from monitoring.alert_manager import AlertLevel, AlertManager, LEVEL_COLORS, logger
from monitoring.behavior_monitor import BehaviorMonitor
from recognition.gesture_recognition import MultiHandGestureRecognizer
from reports.report_generator import ReportGenerator


class Dashboard(ctk.CTk):
    """Main desktop window that controls monitoring, alerts, and reports."""

    def __init__(self):
        ctk.set_appearance_mode(THEME)
        ctk.set_default_color_theme(COLOR_SCHEME)
        super().__init__()

        self.settings = load_settings()
        self.alert_manager = AlertManager(
            cooldown_seconds=self.settings.alert_cooldown_seconds,
            logging_enabled=self.settings.logging_enabled,
        )
        self.behavior_monitor = BehaviorMonitor(self.alert_manager, self.settings)
        self.gesture_recognizer = MultiHandGestureRecognizer(self.settings)
        self.report_generator = ReportGenerator()

        self.face_detector = None
        self.hand_detector = None
        self._capture_thread = None
        self._stop_event = threading.Event()
        self._monitoring = False
        self._closing = False

        self._frame_queue = queue.Queue(maxsize=1)
        self._status_queue = queue.Queue(maxsize=1)
        self._message_queue = queue.Queue()
        self._display_image = None
        self._last_alert_signature = None
        self._poll_job = None
        self._clock_job = None

        self.title(WINDOW_TITLE)
        self.geometry("1320x820")
        self.minsize(1080, 720)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_header()
        self._build_body()
        self._build_footer()
        self._update_clock()
        self._poll_runtime_queues()

    def _build_header(self):
        """Create the title, live clock, and global monitoring status."""
        header = ctk.CTkFrame(self, height=58, corner_radius=0, fg_color="#0f172a")
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        title_area = ctk.CTkFrame(header, fg_color="transparent")
        title_area.pack(side="left", padx=18, pady=8)
        ctk.CTkLabel(
            title_area,
            text="AI Exam Monitoring System",
            font=ctk.CTkFont(size=19, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_area,
            text="Real-time face, hand, gesture, and activity monitoring",
            text_color="#94a3b8",
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w")

        self._clock_label = ctk.CTkLabel(
            header,
            text="",
            text_color="#cbd5e1",
            font=ctk.CTkFont(size=12),
        )
        self._clock_label.pack(side="right", padx=18)

        self._global_badge = StatusBadge(header, "IDLE", COLOR_NEUTRAL)
        self._global_badge.pack(side="right", padx=6)

    def _build_body(self):
        """Create the responsive camera, statistics, and alert panels."""
        body = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=10)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self._build_camera_panel(body)
        self._build_monitoring_panel(body)

    def _build_camera_panel(self, parent):
        """Create the embedded OpenCV feed and current detection badges."""
        camera_panel = ctk.CTkFrame(parent, corner_radius=10)
        camera_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        camera_panel.columnconfigure(0, weight=1)
        camera_panel.rowconfigure(0, weight=1)

        feed_wrapper = ctk.CTkFrame(camera_panel, fg_color="#020617", corner_radius=8)
        feed_wrapper.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))

        self._feed_label = ctk.CTkLabel(
            feed_wrapper,
            text="Camera feed is stopped.\nPress Start Monitoring to begin.",
            text_color="#64748b",
            font=ctk.CTkFont(size=14),
        )
        self._feed_label.pack(fill="both", expand=True, padx=4, pady=4)

        badge_row = ctk.CTkFrame(camera_panel, fg_color="transparent")
        badge_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        badge_row.columnconfigure((0, 1, 2, 3), weight=1)

        self._face_badge = self._create_status_item(
            badge_row, 0, "Face Status", "Waiting"
        )
        self._count_badge = self._create_status_item(
            badge_row, 1, "Face Count", "0"
        )
        self._hand_badge = self._create_status_item(
            badge_row, 2, "Hand Status", "No hands"
        )
        self._gesture_badge = self._create_status_item(
            badge_row, 3, "Gesture", "None"
        )

    def _create_status_item(self, parent, column, title, initial_text):
        """Create one title and value badge used below the camera feed."""
        item_frame = ctk.CTkFrame(parent, fg_color="transparent")
        item_frame.grid(row=0, column=column, padx=4, sticky="ew")
        ctk.CTkLabel(
            item_frame,
            text=title,
            text_color="#94a3b8",
            font=ctk.CTkFont(size=10),
        ).pack()
        badge = StatusBadge(item_frame, initial_text, COLOR_NEUTRAL)
        badge.pack(fill="x")
        return badge

    def _build_monitoring_panel(self, parent):
        """Create session statistics and the scrollable alert history."""
        right_panel = ctk.CTkFrame(parent, corner_radius=10)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(2, weight=1)

        stats_card = ctk.CTkFrame(right_panel, fg_color="#1e293b", corner_radius=8)
        stats_card.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        stats_card.columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            stats_card,
            text="Session Statistics",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(8, 4), sticky="w")

        self._stat_frames = self._create_stat_row(stats_card, 1, "Frames Processed")
        self._stat_violations = self._create_stat_row(stats_card, 2, "Total Violations")
        self._stat_warnings = self._create_stat_row(stats_card, 3, "Warning Alerts")
        self._stat_critical = self._create_stat_row(stats_card, 4, "Critical Alerts")
        self._stat_face = self._create_stat_row(stats_card, 5, "Face Violations")
        self._stat_gesture = self._create_stat_row(stats_card, 6, "Gesture Violations")

        ctk.CTkLabel(
            stats_card,
            text="Current Alert Status",
            text_color="#94a3b8",
            font=ctk.CTkFont(size=11),
        ).grid(row=7, column=0, padx=10, pady=(3, 8), sticky="w")
        self._alert_badge = StatusBadge(stats_card, "IDLE", COLOR_NEUTRAL)
        self._alert_badge.grid(row=7, column=1, padx=10, pady=(3, 8), sticky="e")

        ctk.CTkLabel(
            right_panel,
            text="Alert History",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=1, column=0, padx=14, pady=(4, 3), sticky="w")

        self._alert_scroll = ctk.CTkScrollableFrame(
            right_panel, fg_color="#111827", corner_radius=8
        )
        self._alert_scroll.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._no_alerts_label = ctk.CTkLabel(
            self._alert_scroll,
            text="No alerts have been generated.",
            text_color="#64748b",
            font=ctk.CTkFont(size=12),
        )
        self._no_alerts_label.pack(pady=20)

    def _create_stat_row(self, parent, row, label_text):
        """Create one name and count row in the statistics card."""
        ctk.CTkLabel(
            parent,
            text=label_text,
            text_color="#94a3b8",
            font=ctk.CTkFont(size=11),
        ).grid(row=row, column=0, padx=10, pady=2, sticky="w")
        value_label = ctk.CTkLabel(
            parent,
            text="0",
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        value_label.grid(row=row, column=1, padx=10, pady=2, sticky="e")
        return value_label

    def _build_footer(self):
        """Create monitoring, report, alert, and settings buttons."""
        footer = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color="#0f172a")
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        common_button = {
            "height": 34,
            "corner_radius": 8,
            "font": ctk.CTkFont(size=12),
        }

        self._start_button = ctk.CTkButton(
            footer,
            text="Start Monitoring",
            command=self._start_monitoring,
            fg_color=COLOR_OK,
            hover_color="#16a34a",
            **common_button,
        )
        self._start_button.pack(side="left", padx=(14, 5), pady=13)

        self._stop_button = ctk.CTkButton(
            footer,
            text="Stop",
            command=self._stop_monitoring,
            fg_color=COLOR_CRITICAL,
            hover_color="#dc2626",
            state="disabled",
            **common_button,
        )
        self._stop_button.pack(side="left", padx=5, pady=13)

        ctk.CTkButton(
            footer,
            text="Export CSV",
            command=self._export_csv,
            fg_color=COLOR_INFO,
            hover_color="#2563eb",
            **common_button,
        ).pack(side="left", padx=5, pady=13)

        ctk.CTkButton(
            footer,
            text="Export PDF",
            command=self._export_pdf,
            fg_color="#6366f1",
            hover_color="#4f46e5",
            **common_button,
        ).pack(side="left", padx=5, pady=13)

        ctk.CTkButton(
            footer,
            text="Clear Panel",
            command=self._clear_alert_panel,
            fg_color=COLOR_NEUTRAL,
            hover_color="#475569",
            **common_button,
        ).pack(side="left", padx=5, pady=13)

        ctk.CTkButton(
            footer,
            text="Settings",
            command=self._open_settings,
            fg_color="#334155",
            hover_color="#1e293b",
            **common_button,
        ).pack(side="right", padx=14, pady=13)

    def _start_monitoring(self):
        """Create fresh detectors and start the webcam worker thread."""
        if self._monitoring:
            return
        if self._capture_thread and self._capture_thread.is_alive():
            messagebox.showwarning(
                "Camera Is Stopping",
                "Please wait a moment for the previous camera session to close.",
            )
            return

        try:
            self._release_detectors()
            self.face_detector = FaceDetector(self.settings)
            self.hand_detector = HandDetector(self.settings)
        except Exception as error:
            logger.exception("Detection models could not be created.")
            messagebox.showerror(
                "Detection Setup Error",
                "The AI detection models could not start.\n\n"
                f"Details: {error}",
            )
            return

        self.behavior_monitor.reset_session()
        self.gesture_recognizer.reset()
        self._stop_event.clear()
        self._monitoring = True
        self._start_button.configure(state="disabled")
        self._stop_button.configure(state="normal")
        self._global_badge.set_status("STARTING", COLOR_INFO)
        self._alert_badge.set_status("STARTING", COLOR_INFO)

        self.alert_manager.info(
            "MONITORING_STARTED", "Monitoring session started.", force=True
        )
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="ExamMonitoringCapture",
            daemon=True,
        )
        self._capture_thread.start()

    def _capture_loop(self):
        """Read camera frames and run all detection modules in the background."""
        camera = None
        try:
            camera = cv2.VideoCapture(self.settings.camera_index)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)
            camera.set(cv2.CAP_PROP_FPS, self.settings.camera_fps)

            if not camera.isOpened():
                raise RuntimeError(
                    "The webcam could not be opened. Close other camera apps and check the camera index."
                )

            failed_reads = 0
            while not self._stop_event.is_set():
                frame_was_read, frame = camera.read()
                if not frame_was_read or frame is None:
                    failed_reads += 1
                    if failed_reads >= 30:
                        raise RuntimeError("The webcam stopped returning video frames.")
                    time.sleep(0.03)
                    continue

                failed_reads = 0
                if self.settings.mirror_camera:
                    frame = cv2.flip(frame, 1)

                detection_frame, quality_results = prepare_detection_frame(
                    frame, self.settings
                )
                face_frame, face_results = self.face_detector.process_frame(
                    detection_frame
                )
                face_results.update(quality_results)
                annotated_frame, hand_results = self.hand_detector.process_frame(
                    detection_frame, face_frame
                )

                landmark_lists = [
                    self.hand_detector.get_landmark_list(
                        index, detection_frame.shape
                    )
                    for index in range(hand_results["hand_count"])
                ]
                gesture_results = self.gesture_recognizer.recognize_all(
                    landmark_lists, hand_results["hand_labels"]
                )
                self.behavior_monitor.analyse(
                    face_results, hand_results, gesture_results
                )
                draw_monitoring_overlay(
                    annotated_frame,
                    face_results["face_count"],
                    gesture_results,
                )

                status_snapshot = self.behavior_monitor.get_snapshot()
                status_snapshot["face_count"] = face_results["face_count"]
                status_snapshot["hand_count"] = hand_results["hand_count"]
                self._replace_queue_item(self._frame_queue, annotated_frame)
                self._replace_queue_item(self._status_queue, status_snapshot)

        except Exception as error:
            logger.exception("Monitoring stopped because of a runtime error.")
            self._message_queue.put(("error", str(error)))
        finally:
            if camera is not None:
                camera.release()
            self._message_queue.put(("stopped", None))

    def _replace_queue_item(self, target_queue, value):
        """Keep only the newest frame or status object in a small queue."""
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            target_queue.put_nowait(value)
        except queue.Full:
            pass

    def _poll_runtime_queues(self):
        """Move worker results into Tkinter widgets on the main GUI thread."""
        if self._closing:
            return

        self._process_worker_messages()
        latest_frame = self._get_latest_queue_item(self._frame_queue)
        if latest_frame is not None:
            self._show_frame(latest_frame)

        latest_status = self._get_latest_queue_item(self._status_queue)
        if latest_status is not None:
            self._show_status(latest_status)

        self._refresh_alert_history()
        self._poll_job = self.after(50, self._poll_runtime_queues)

    def _get_latest_queue_item(self, target_queue):
        """Read the newest available queue item without waiting."""
        latest_item = None
        while True:
            try:
                latest_item = target_queue.get_nowait()
            except queue.Empty:
                return latest_item

    def _process_worker_messages(self):
        """Display camera errors and restore controls after a worker stops."""
        while True:
            try:
                message_type, message_text = self._message_queue.get_nowait()
            except queue.Empty:
                return

            if message_type == "error" and not self._closing:
                messagebox.showerror("Monitoring Error", message_text)
            elif message_type == "stopped":
                self._finish_stopped_state()

    def _show_frame(self, frame):
        """Resize and display the newest OpenCV frame inside CustomTkinter."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        available_width = max(320, self._feed_label.winfo_width() - 12)
        available_height = max(240, self._feed_label.winfo_height() - 12)
        target_width = min(FEED_WIDTH, available_width)
        target_height = min(FEED_HEIGHT, available_height)
        image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

        self._display_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size,
        )
        self._feed_label.configure(image=self._display_image, text="")

    def _show_status(self, snapshot):
        """Update badges and session counts from one monitoring snapshot."""
        face_count = snapshot["face_count"]
        face_color = COLOR_OK if face_count == 1 else COLOR_CRITICAL
        if snapshot["face_status"] == "Near frame edge":
            face_color = COLOR_WARNING

        self._face_badge.set_status(snapshot["face_status"], face_color)
        self._count_badge.set_status(
            str(face_count), COLOR_OK if face_count == 1 else COLOR_WARNING
        )
        self._hand_badge.set_status(
            snapshot["hand_status"],
            COLOR_INFO if snapshot["hand_count"] else COLOR_NEUTRAL,
        )

        gesture_text = snapshot["gesture_status"]
        gesture_color = (
            COLOR_WARNING
            if "Phone Gesture" in gesture_text or "Victory Sign" in gesture_text
            else COLOR_INFO
        )
        self._gesture_badge.set_status(gesture_text[:28], gesture_color)

        alert_color = LEVEL_COLORS.get(snapshot["alert_level"], COLOR_NEUTRAL)
        self._alert_badge.set_status(snapshot["alert_status"], alert_color)
        self._global_badge.set_status(snapshot["alert_status"], alert_color)

        stats = snapshot["stats"]
        alert_summary = self.alert_manager.summary()
        face_violations = (
            stats["face_missing_violations"]
            + stats["multiple_face_violations"]
            + stats["face_outside_violations"]
            + stats["look_away_violations"]
            + stats["hand_cover_violations"]
        )
        self._stat_frames.configure(text=str(stats["frames_processed"]))
        self._stat_violations.configure(text=str(stats["total_violations"]))
        self._stat_warnings.configure(
            text=str(alert_summary.get(AlertLevel.WARNING, 0))
        )
        self._stat_critical.configure(
            text=str(alert_summary.get(AlertLevel.CRITICAL, 0))
        )
        self._stat_face.configure(text=str(face_violations))
        self._stat_gesture.configure(
            text=str(stats["suspicious_gesture_violations"])
        )

    def _refresh_alert_history(self):
        """Rebuild the alert list only when the alert history changes."""
        alert_history = self.alert_manager.get_history()
        signature = (
            len(alert_history),
            alert_history[-1].timestamp if alert_history else None,
        )
        if signature == self._last_alert_signature:
            return
        self._last_alert_signature = signature

        for widget in self._alert_scroll.winfo_children():
            widget.destroy()

        if not alert_history:
            self._no_alerts_label = ctk.CTkLabel(
                self._alert_scroll,
                text="No alerts have been generated.",
                text_color="#64748b",
                font=ctk.CTkFont(size=12),
            )
            self._no_alerts_label.pack(pady=20)
            return

        for alert in alert_history[-100:]:
            alert_row = AlertRow(self._alert_scroll, alert)
            alert_row.pack(fill="x", padx=3, pady=2)

    def _stop_monitoring(self):
        """Ask the webcam worker to stop and wait briefly for cleanup."""
        if not self._monitoring and not (
            self._capture_thread and self._capture_thread.is_alive()
        ):
            return

        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.5)

        self.alert_manager.info(
            "MONITORING_STOPPED", "Monitoring session stopped.", force=True
        )
        if self._capture_thread and self._capture_thread.is_alive():
            self._global_badge.set_status("STOPPING", COLOR_WARNING)
            self._alert_badge.set_status("STOPPING", COLOR_WARNING)
            self._start_button.configure(state="disabled")
            self._stop_button.configure(state="disabled")
        else:
            self._finish_stopped_state()

    def _finish_stopped_state(self):
        """Restore stopped controls after normal completion or a camera error."""
        self._monitoring = False
        self._start_button.configure(state="normal")
        self._stop_button.configure(state="disabled")
        self._global_badge.set_status("STOPPED", COLOR_NEUTRAL)
        self._alert_badge.set_status("STOPPED", COLOR_NEUTRAL)

    def _release_detectors(self):
        """Close MediaPipe resources after the capture thread has ended."""
        if self._capture_thread and self._capture_thread.is_alive():
            return
        if self.face_detector is not None:
            self.face_detector.release()
            self.face_detector = None
        if self.hand_detector is not None:
            self.hand_detector.release()
            self.hand_detector = None

    def _export_csv(self):
        """Generate today's CSV report and show the saved file path."""
        try:
            report_path = self.report_generator.export_csv()
            messagebox.showinfo("CSV Report Saved", f"Report saved to:\n{report_path}")
        except Exception as error:
            logger.exception("CSV report export failed.")
            messagebox.showerror("CSV Export Error", str(error))

    def _export_pdf(self):
        """Generate today's PDF report and show the saved file path."""
        try:
            report_path = self.report_generator.export_pdf()
            messagebox.showinfo("PDF Report Saved", f"Report saved to:\n{report_path}")
        except Exception as error:
            logger.exception("PDF report export failed.")
            messagebox.showerror("PDF Export Error", str(error))

    def _clear_alert_panel(self):
        """Clear the visible session history while keeping the CSV audit log."""
        self.alert_manager.clear_history()
        self._last_alert_signature = None
        self._refresh_alert_history()

    def _open_settings(self):
        """Open the modal settings window."""
        SettingsDialog(self, self.settings)

    def apply_settings(self, updated_settings):
        """Stop monitoring, save settings, and apply them to future sessions."""
        if self._monitoring:
            self._stop_monitoring()

        self._release_detectors()
        self.settings = updated_settings.validate()
        save_settings(self.settings)
        self.alert_manager.cooldown_seconds = self.settings.alert_cooldown_seconds
        self.alert_manager.set_logging_enabled(self.settings.logging_enabled)
        self.behavior_monitor.settings = self.settings
        self.gesture_recognizer = MultiHandGestureRecognizer(self.settings)
        messagebox.showinfo(
            "Settings Saved",
            "Settings were saved. Start monitoring to use the new values.",
        )

    def _update_clock(self):
        """Update the header clock once every second."""
        self._clock_label.configure(
            text=datetime.now().strftime("%A, %d %B %Y  %H:%M:%S")
        )
        self._clock_job = self.after(1000, self._update_clock)

    def _on_close(self):
        """Stop background work and close the desktop window safely."""
        self._closing = True
        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._release_detectors()

        if self._poll_job:
            self.after_cancel(self._poll_job)
        if self._clock_job:
            self.after_cancel(self._clock_job)
        self.destroy()
