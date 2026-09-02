import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prepare_data import split_grouped_paths


class GroupSplitTests(unittest.TestCase):
    def test_related_images_remain_in_exactly_one_split(self):
        records = [(Path(f"image-{group}-{image}.jpg"), f"group-{group}") for group in range(10) for image in range(2)]
        train, validation, test = split_grouped_paths(records, 0.15, 0.15, seed=42)
        split_groups = [{path.stem.rsplit("-", 1)[0] for path in split} for split in (train, validation, test)]
        self.assertFalse(split_groups[0] & split_groups[1])
        self.assertFalse(split_groups[0] & split_groups[2])
        self.assertFalse(split_groups[1] & split_groups[2])
        self.assertEqual(len(train) + len(validation) + len(test), len(records))


if __name__ == "__main__":
    unittest.main()
