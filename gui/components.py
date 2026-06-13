"""Small reusable CustomTkinter widgets used by the dashboard."""

import customtkinter as ctk

from config import COLOR_NEUTRAL
from monitoring.alert_manager import LEVEL_COLORS


class StatusBadge(ctk.CTkLabel):
    """Show a short status value inside a colored rounded label."""

    def __init__(self, master, text="None", color=COLOR_NEUTRAL, **kwargs):
        super().__init__(
            master,
            text=text,
            fg_color=color,
            corner_radius=8,
            text_color="white",
            font=ctk.CTkFont(size=12, weight="bold"),
            padx=10,
            pady=4,
            **kwargs,
        )

    def set_status(self, text, color):
        """Update the badge text and color together."""
        self.configure(text=text, fg_color=color)


class AlertRow(ctk.CTkFrame):
    """Display one monitoring alert inside the history panel."""

    def __init__(self, master, alert):
        super().__init__(master, corner_radius=6, fg_color="#1e293b")
        level_color = LEVEL_COLORS.get(alert.level, COLOR_NEUTRAL)

        ctk.CTkLabel(
            self,
            text=alert.level,
            width=72,
            fg_color=level_color,
            corner_radius=6,
            text_color="white",
            font=ctk.CTkFont(size=10, weight="bold"),
        ).grid(row=0, column=0, rowspan=2, padx=6, pady=6, sticky="ns")

        ctk.CTkLabel(
            self,
            text=f"{alert.timestamp:%H:%M:%S}  {alert.event_type.replace('_', ' ').title()}",
            text_color="#94a3b8",
            font=ctk.CTkFont(size=10),
            anchor="w",
        ).grid(row=0, column=1, padx=(0, 6), pady=(5, 0), sticky="ew")

        ctk.CTkLabel(
            self,
            text=alert.description,
            font=ctk.CTkFont(size=11),
            anchor="w",
            justify="left",
            wraplength=370,
        ).grid(row=1, column=1, padx=(0, 6), pady=(0, 5), sticky="ew")
        self.columnconfigure(1, weight=1)
