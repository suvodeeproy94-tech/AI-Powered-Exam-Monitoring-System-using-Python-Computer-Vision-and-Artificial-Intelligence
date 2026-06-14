"""Modal window for camera, detection, alert, and logging settings."""

from dataclasses import replace
import tkinter as tk

import customtkinter as ctk

from config import COLOR_NEUTRAL


class SettingsDialog(ctk.CTkToplevel):
    """Edit camera, sensitivity, alert, and logging settings."""

    RESOLUTIONS = {
        "640 x 480": (640, 480),
        "960 x 540": (960, 540),
        "1280 x 720": (1280, 720),
    }

    def __init__(self, parent, current_settings):
        super().__init__(parent)
        self.parent = parent
        self.updated_settings = replace(current_settings)
        self.title("Monitoring Settings")
        self.geometry("540x650")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self._build_form()

    def _build_form(self):
        """Create all settings fields with simple labels and help text."""
        container = ctk.CTkScrollableFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=12, pady=12)

        ctk.CTkLabel(
            container,
            text="Camera and Detection Settings",
            font=ctk.CTkFont(size=17, weight="bold"),
        ).pack(anchor="w", padx=8, pady=(4, 12))

        self.camera_index_var = tk.IntVar(value=self.updated_settings.camera_index)
        self._add_slider(
            container, "Camera Index", self.camera_index_var, 0, 4, 4, integer=True
        )

        self.resolution_var = tk.StringVar(value=self._find_resolution_name())
        self._add_option(
            container,
            "Camera Resolution",
            self.resolution_var,
            list(self.RESOLUTIONS),
        )

        self.face_confidence_var = tk.DoubleVar(
            value=self.updated_settings.face_detection_confidence
        )
        self._add_slider(
            container,
            "Face Detection Confidence",
            self.face_confidence_var,
            0.30,
            0.90,
            12,
        )

        self.hand_confidence_var = tk.DoubleVar(
            value=self.updated_settings.hand_detection_confidence
        )
        self._add_slider(
            container,
            "Hand Detection Confidence",
            self.hand_confidence_var,
            0.30,
            0.90,
            12,
        )

        self.face_missing_seconds_var = tk.DoubleVar(
            value=self.updated_settings.face_missing_seconds
        )
        self._add_slider(
            container,
            "Missing Face Alert Time",
            self.face_missing_seconds_var,
            0.5,
            5,
            18,
        )

        self.calibration_seconds_var = tk.DoubleVar(
            value=self.updated_settings.calibration_seconds
        )
        self._add_slider(
            container,
            "Calibration Time (seconds)",
            self.calibration_seconds_var,
            2,
            15,
            13,
        )

        self.look_away_seconds_var = tk.DoubleVar(
            value=self.updated_settings.look_away_seconds
        )
        self._add_slider(
            container,
            "Look-Away Alert Time",
            self.look_away_seconds_var,
            0.5,
            5,
            18,
        )

        self.cooldown_var = tk.DoubleVar(
            value=self.updated_settings.alert_cooldown_seconds
        )
        self._add_slider(
            container,
            "Repeated Alert Cooldown (seconds)",
            self.cooldown_var,
            0,
            30,
            30,
        )

        self.mirror_var = tk.BooleanVar(value=self.updated_settings.mirror_camera)
        self._add_switch(container, "Mirror Camera Preview", self.mirror_var)

        self.mesh_var = tk.BooleanVar(value=self.updated_settings.draw_face_mesh)
        self._add_switch(container, "Draw Face Landmarks", self.mesh_var)

        self.enhance_var = tk.BooleanVar(
            value=self.updated_settings.enhance_low_light
        )
        self._add_switch(
            container, "Improve Detection in Low Light", self.enhance_var
        )

        self.calibration_var = tk.BooleanVar(
            value=self.updated_settings.calibration_enabled
        )
        self._add_switch(
            container, "Run Personal Calibration", self.calibration_var
        )

        self.head_pose_var = tk.BooleanVar(
            value=self.updated_settings.head_pose_enabled
        )
        self._add_switch(container, "Use Head Pose Detection", self.head_pose_var)

        self.gaze_var = tk.BooleanVar(
            value=self.updated_settings.gaze_tracking_enabled
        )
        self._add_switch(container, "Use Iris and Gaze Tracking", self.gaze_var)

        self.evidence_var = tk.BooleanVar(
            value=self.updated_settings.evidence_capture_enabled
        )
        self._add_switch(
            container, "Save Alert Evidence Images", self.evidence_var
        )

        self.trained_model_var = tk.BooleanVar(
            value=self.updated_settings.trained_gesture_model_enabled
        )
        self._add_switch(
            container, "Use Trained Gesture Model When Available", self.trained_model_var
        )

        self.logging_var = tk.BooleanVar(value=self.updated_settings.logging_enabled)
        self._add_switch(container, "Save Activity Logs", self.logging_var)

        button_row = ctk.CTkFrame(container, fg_color="transparent")
        button_row.pack(fill="x", padx=8, pady=(18, 6))
        ctk.CTkButton(
            button_row,
            text="Cancel",
            command=self.destroy,
            fg_color=COLOR_NEUTRAL,
            hover_color="#475569",
        ).pack(side="right", padx=(6, 0))
        ctk.CTkButton(
            button_row,
            text="Save Settings",
            command=self._save,
        ).pack(side="right")

    def _find_resolution_name(self):
        """Find the menu label matching the current width and height."""
        current_size = (
            self.updated_settings.camera_width,
            self.updated_settings.camera_height,
        )
        for resolution_name, resolution_size in self.RESOLUTIONS.items():
            if resolution_size == current_size:
                return resolution_name
        return "640 x 480"

    def _add_slider(
        self,
        parent,
        label_text,
        variable,
        minimum,
        maximum,
        steps,
        integer=False,
    ):
        """Add a labeled slider with its current value shown on the right."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=7)
        row.columnconfigure(1, weight=1)

        ctk.CTkLabel(row, text=label_text, width=220, anchor="w").grid(
            row=0, column=0, sticky="w"
        )
        value_label = ctk.CTkLabel(row, text="", width=58, anchor="e")
        value_label.grid(row=0, column=2, sticky="e")

        def update_value(display_value):
            shown_value = (
                int(float(display_value))
                if integer
                else round(float(display_value), 2)
            )
            value_label.configure(text=str(shown_value))

        slider = ctk.CTkSlider(
            row,
            variable=variable,
            from_=minimum,
            to=maximum,
            number_of_steps=steps,
            command=update_value,
        )
        slider.grid(row=0, column=1, padx=10, sticky="ew")
        update_value(variable.get())

    def _add_option(self, parent, label_text, variable, values):
        """Add a labeled option menu."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=7)
        ctk.CTkLabel(row, text=label_text, width=220, anchor="w").pack(side="left")
        ctk.CTkOptionMenu(row, variable=variable, values=values).pack(side="right")

    def _add_switch(self, parent, label_text, variable):
        """Add one on/off setting switch."""
        ctk.CTkSwitch(parent, text=label_text, variable=variable).pack(
            fill="x", padx=8, pady=8
        )

    def _save(self):
        """Copy form values into AppSettings and send them to the dashboard."""
        width, height = self.RESOLUTIONS[self.resolution_var.get()]
        self.updated_settings.camera_index = int(self.camera_index_var.get())
        self.updated_settings.camera_width = width
        self.updated_settings.camera_height = height
        self.updated_settings.face_detection_confidence = round(
            self.face_confidence_var.get(), 2
        )
        self.updated_settings.hand_detection_confidence = round(
            self.hand_confidence_var.get(), 2
        )
        self.updated_settings.face_missing_seconds = round(
            self.face_missing_seconds_var.get(), 1
        )
        self.updated_settings.calibration_seconds = round(
            self.calibration_seconds_var.get(), 1
        )
        self.updated_settings.look_away_seconds = round(
            self.look_away_seconds_var.get(), 1
        )
        self.updated_settings.alert_cooldown_seconds = round(
            self.cooldown_var.get(), 1
        )
        self.updated_settings.mirror_camera = bool(self.mirror_var.get())
        self.updated_settings.draw_face_mesh = bool(self.mesh_var.get())
        self.updated_settings.enhance_low_light = bool(self.enhance_var.get())
        self.updated_settings.calibration_enabled = bool(self.calibration_var.get())
        self.updated_settings.head_pose_enabled = bool(self.head_pose_var.get())
        self.updated_settings.gaze_tracking_enabled = bool(self.gaze_var.get())
        self.updated_settings.evidence_capture_enabled = bool(self.evidence_var.get())
        self.updated_settings.trained_gesture_model_enabled = bool(
            self.trained_model_var.get()
        )
        self.updated_settings.logging_enabled = bool(self.logging_var.get())
        self.parent.apply_settings(self.updated_settings)
        self.destroy()
