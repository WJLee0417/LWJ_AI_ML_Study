import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import parse_args


class ConfigTests(unittest.TestCase):
    def test_yaml_experiment_values_and_cli_override(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "experiment.yaml"
            config.write_text(yaml.safe_dump({"model": "scratch", "augmentation": "off", "epochs": 3}), encoding="utf-8")
            with patch.object(sys, "argv", ["train.py", "--config", str(config), "--epochs", "2"]):
                args = parse_args()
        self.assertEqual(args.model, "scratch")
        self.assertEqual(args.augmentation, "off")
        self.assertEqual(args.epochs, 2)


if __name__ == "__main__":
    unittest.main()
