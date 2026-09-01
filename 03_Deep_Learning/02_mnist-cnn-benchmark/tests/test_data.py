import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import stratified_indices


class DataTests(unittest.TestCase):
    def test_stratified_split_is_reproducible_and_balanced(self):
        labels = np.repeat(np.arange(10), 20)
        train_indices, validation_indices = stratified_indices(labels, 0.2, seed=42)
        repeated_train, repeated_validation = stratified_indices(labels, 0.2, seed=42)

        self.assertEqual((train_indices, validation_indices), (repeated_train, repeated_validation))
        self.assertEqual(len(train_indices), 160)
        self.assertEqual(len(validation_indices), 40)
        self.assertEqual(set(train_indices) & set(validation_indices), set())
        self.assertTrue(np.all(np.bincount(labels[validation_indices]) == 4))

    def test_invalid_validation_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            stratified_indices([0, 1, 0, 1], 1.0, seed=42)


if __name__ == "__main__":
    unittest.main()
