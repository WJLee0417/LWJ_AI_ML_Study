import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration import choose_review_threshold, expected_calibration_error


class CalibrationTests(unittest.TestCase):
    def test_review_threshold_maximizes_eligible_automatic_volume(self):
        labels = np.array([0, 1, 0, 1])
        probabilities = np.array([[0.99, 0.01], [0.09, 0.91], [0.88, 0.12], [0.82, 0.18]])
        policy = choose_review_threshold(labels, probabilities, minimum_precision=0.9)
        self.assertEqual(policy["automatic_count"], 3)
        self.assertEqual(policy["review_threshold"], 0.88)

    def test_ece_is_zero_for_perfectly_calibrated_single_predictions(self):
        probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(expected_calibration_error(probabilities, np.array([0, 1])), 0.0)


if __name__ == "__main__":
    unittest.main()
