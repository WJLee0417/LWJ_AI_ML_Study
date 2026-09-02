import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import temporal_split


class TemporalSplitTests(unittest.TestCase):
    def test_temporal_split_keeps_future_rows_out_of_training(self):
        frame = pd.DataFrame(
            {
                "text": [f"문의 {index}" for index in range(12)],
                "label": ["delivery", "refund"] * 6,
                "timestamp": pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC"),
            }
        )
        train, validation, test = temporal_split(frame, 0.25, 0.25)
        self.assertEqual(len(train) + len(validation) + len(test), len(frame))
        self.assertLessEqual(train["timestamp"].max(), validation["timestamp"].min())
        self.assertLessEqual(validation["timestamp"].max(), test["timestamp"].min())
        self.assertTrue(set(test["label"]).issubset(set(train["label"])))


if __name__ == "__main__":
    unittest.main()
