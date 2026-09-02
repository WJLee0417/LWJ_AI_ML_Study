import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import parse_args


class ConfigTests(unittest.TestCase):
    def test_yaml_values_load_and_cli_overrides_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            path.write_text(yaml.safe_dump({"epochs": 3, "seed": 7, "model": "cnn"}), encoding="utf-8")
            with patch.object(sys, "argv", ["train.py", "--config", str(path), "--epochs", "2"]):
                args = parse_args()
        self.assertEqual(args.model, "cnn")
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.epochs, 2)


if __name__ == "__main__":
    unittest.main()
