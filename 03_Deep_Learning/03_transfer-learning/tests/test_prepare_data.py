import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prepare_data import split_paths


class SplitTests(unittest.TestCase):
    def test_split_has_no_overlap_and_is_reproducible(self):
        paths = [Path(f"image-{index}.jpg") for index in range(20)]
        first = split_paths(paths, 0.2, 0.2, seed=42)
        second = split_paths(paths, 0.2, 0.2, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first[0]) & set(first[1])), 0)
        self.assertEqual(len(set(first[0]) & set(first[2])), 0)
        self.assertEqual(len(set(first[1]) & set(first[2])), 0)
        self.assertEqual(sum(map(len, first)), 20)


if __name__ == "__main__":
    unittest.main()
