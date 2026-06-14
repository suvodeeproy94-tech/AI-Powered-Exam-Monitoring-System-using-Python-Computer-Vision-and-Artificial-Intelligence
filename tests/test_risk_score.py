"""Tests for explainable suspicion scoring and normal-behavior decay."""

import unittest

from config import AppSettings
from monitoring.risk_score import RiskScoreEngine


class RiskScoreTests(unittest.TestCase):
    """Verify event weights, severity levels, and time decay."""

    def test_multiple_faces_creates_warning_score(self):
        settings = AppSettings(risk_warning_threshold=35, risk_critical_threshold=70)
        engine = RiskScoreEngine(settings)
        engine.reset(current_time=0.0)

        engine.add_event("MULTIPLE_FACES")

        self.assertEqual(engine.score, 50.0)
        self.assertEqual(engine.level(), "WARNING")

    def test_score_decreases_during_normal_time(self):
        settings = AppSettings(risk_decay_per_second=2.0)
        engine = RiskScoreEngine(settings)
        engine.reset(current_time=0.0)
        engine.add_event("FACE_MISSING")

        engine.update_time(current_time=5.0)

        self.assertEqual(engine.score, 20.0)


if __name__ == "__main__":
    unittest.main()
