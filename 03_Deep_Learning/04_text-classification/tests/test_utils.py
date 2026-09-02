import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import upsert_csv


class CsvUpsertTests(unittest.TestCase):
    def test_preserves_columns_from_different_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"
            upsert_csv(path, {"experiment": "baseline", "macro_f1": 0.8})
            upsert_csv(path, {"experiment": "transformer", "macro_f1": 0.9, "pretrained_model": "klue/bert-base"})
            with path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["pretrained_model"], "klue/bert-base")


if __name__ == "__main__":
    unittest.main()
