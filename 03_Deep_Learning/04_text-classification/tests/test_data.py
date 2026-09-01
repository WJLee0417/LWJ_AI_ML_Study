import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import stratified_split


class SplitTests(unittest.TestCase):
    def test_stratified_split_is_reproducible_and_exhaustive(self):
        frame = pd.DataFrame({"text": [f"문의 {index}" for index in range(20)], "label": ["배송"] * 10 + ["환불"] * 10})
        first = stratified_split(frame, 0.2, 0.2, 42)
        second = stratified_split(frame, 0.2, 0.2, 42)
        self.assertEqual([split["text"].tolist() for split in first], [split["text"].tolist() for split in second])
        self.assertEqual(sum(len(split) for split in first), len(frame))
        self.assertEqual(set(first[0]["text"]) & set(first[2]["text"]), set())


if __name__ == "__main__":
    unittest.main()
