"""Tests for optional alert evidence image storage."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import AppSettings
from monitoring.evidence_manager import EvidenceManager


class EvidenceManagerTests(unittest.TestCase):
    """Verify evidence is saved only for configured alert levels."""

    def test_warning_frame_is_saved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = EvidenceManager(
                AppSettings(evidence_capture_enabled=True),
                Path(temporary_directory),
            )
            frame = np.full((80, 100, 3), 120, dtype=np.uint8)

            evidence_path = manager.capture(frame, "LOOKING_AWAY", "WARNING")

            self.assertTrue(Path(evidence_path).exists())

    def test_information_frame_is_not_saved_by_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = EvidenceManager(
                AppSettings(evidence_capture_enabled=True),
                Path(temporary_directory),
            )
            frame = np.zeros((80, 100, 3), dtype=np.uint8)

            evidence_path = manager.capture(frame, "STARTED", "INFO")

            self.assertEqual(evidence_path, "")


if __name__ == "__main__":
    unittest.main()
