import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy import apply_review_policy, choose_review_threshold


class PolicyTests(unittest.TestCase):
    def test_threshold_meets_precision_target_with_largest_eligible_volume(self):
        actual = np.array([0, 1, 0, 1])
        predicted = np.array([0, 1, 0, 0])
        confidence = np.array([0.99, 0.91, 0.88, 0.82])
        policy = choose_review_threshold(actual, predicted, confidence, minimum_precision=0.9)
        self.assertEqual(policy["automated_count"], 3)
        self.assertEqual(policy["threshold"], 0.88)
        self.assertTrue(np.array_equal(apply_review_policy(confidence, 0.88), np.array([False, False, False, True])))


if __name__ == "__main__":
    unittest.main()
