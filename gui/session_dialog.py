"""Dialog for collecting student and exam details before monitoring starts.

The monitoring system needs these details so logs, evidence, and reports can be
linked to one real exam session.
"""

import customtkinter as ctk


class SessionDetailsDialog(ctk.CTkToplevel):
    """Ask the user for simple student and exam information."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Exam Session Details")
        self.geometry("460x470")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result = None
        self._error_label = None
        self._entries = {}

        self._build_form()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _build_form(self):
        """Create labels, input boxes, and action buttons."""
        frame = ctk.CTkFrame(self, corner_radius=10)
        frame.pack(fill="both", expand=True, padx=16, pady=16)

        ctk.CTkLabel(
            frame,
            text="Enter Exam Session Details",
            font=ctk.CTkFont(size=17, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(12, 4))

        ctk.CTkLabel(
            frame,
            text="These details will be stored with alerts and reports.",
            text_color="#94a3b8",
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=16, pady=(0, 10))

        fields = (
            ("student_name", "Student Name"),
            ("roll_number", "Roll Number"),
            ("exam_name", "Exam Name"),
            ("subject_name", "Subject Name"),
        )
        for field_name, label_text in fields:
            ctk.CTkLabel(
                frame,
                text=label_text,
                text_color="#cbd5e1",
                font=ctk.CTkFont(size=11),
            ).pack(anchor="w", padx=16, pady=(5, 2))
            entry = ctk.CTkEntry(frame, height=34)
            entry.pack(fill="x", padx=16)
            self._entries[field_name] = entry

        self._error_label = ctk.CTkLabel(
            frame,
            text=" ",
            text_color="#ef4444",
            font=ctk.CTkFont(size=11),
        )
        self._error_label.pack(anchor="w", padx=16, pady=(8, 0))

        button_row = ctk.CTkFrame(frame, fg_color="transparent")
        button_row.pack(fill="x", padx=16, pady=(10, 12))

        ctk.CTkButton(
            button_row,
            text="Cancel",
            command=self._cancel,
            fg_color="#64748b",
            hover_color="#475569",
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            button_row,
            text="Start Session",
            command=self._submit,
            fg_color="#22c55e",
            hover_color="#16a34a",
        ).pack(side="right")

        self._entries["student_name"].focus_set()

    def _submit(self):
        """Validate required fields and close the dialog with a result."""
        values = {
            field_name: entry.get().strip()
            for field_name, entry in self._entries.items()
        }
        missing_labels = [
            label
            for field_name, label in (
                ("student_name", "Student Name"),
                ("roll_number", "Roll Number"),
                ("exam_name", "Exam Name"),
                ("subject_name", "Subject Name"),
            )
            if not values[field_name]
        ]
        if missing_labels:
            self._error_label.configure(
                text="Please fill: " + ", ".join(missing_labels)
            )
            return

        self.result = values
        self.destroy()

    def _cancel(self):
        """Close the dialog without starting a monitoring session."""
        self.result = None
        self.destroy()
