import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import apply_temperature, expected_calibration_error
from src.policy import operational_metrics


class CalibrationTests(unittest.TestCase):
    def test_temperature_preserves_probability_rows(self):
        probabilities = np.array([[0.9, 0.1], [0.4, 0.6]])
        scaled = apply_temperature(probabilities, 1.5)
        self.assertTrue(np.allclose(scaled.sum(axis=1), 1.0))
        self.assertTrue(np.array_equal(scaled.argmax(axis=1), probabilities.argmax(axis=1)))

    def test_operational_metrics_include_review_rate(self):
        metrics = operational_metrics(
            np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([0.95, 0.8, 0.7]), 0.8
        )
        self.assertEqual(metrics["automated_count"], 2)
        self.assertEqual(metrics["review_count"], 1)
        self.assertAlmostEqual(metrics["automated_precision"], 0.5)
        self.assertAlmostEqual(metrics["review_rate"], 1 / 3)
        self.assertGreaterEqual(expected_calibration_error(np.array([[0.9, 0.1]]), np.array([0])), 0.0)


if __name__ == "__main__":
    unittest.main()
